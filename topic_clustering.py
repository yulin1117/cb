import os
import re
import json
import torch
import numpy as np
import umap
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sentence_transformers import SentenceTransformer
import hdbscan
from typing import Any

from llm.topic_gpt import TopicGPT
from openalex import load_topic_title_abstract

# -----------------------------
# Helpers: cleaning + stopwords
# -----------------------------
DEFAULT_DOMAIN_STOPWORDS = {
    # generic scientific boilerplate
    "paper", "study", "studies", "result", "results", "method", "methods", "approach", "approaches",
    "propose", "proposed", "novel", "new", "based", "show", "shows", "shown", "demonstrate",
    "experimental", "experiments", "evaluation", "evaluate", "evaluated", "performance",
    "state-of-the-art", "sota", "baseline", "baselines", "framework", "system", "model", "models",
    "dataset", "datasets", "data", "task", "tasks", "analysis", "analyses", "using", "use", "used",
    "we", "our", "ours", "this", "these", "those", "their", "there", "here",
}

def _normalize_text(s: str) -> str:
    """
    Light normalization geared toward scientific abstracts:
    - lowercasing
    - unify hyphens
    - collapse whitespace
    """
    s = s.strip()
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _build_stopwords(extra_stopwords=None):
    sw = set(DEFAULT_DOMAIN_STOPWORDS)
    if extra_stopwords:
        sw |= set(extra_stopwords)
    return sw

def _representative_docs(texts, emb_norm, labels, top_m=5):
    """
    Pick docs closest to centroid (cosine) for each cluster.
    emb_norm should be L2-normalized.
    """
    reps = {}
    unique = sorted(set(labels))
    for lbl in unique:
        if lbl == -1:
            continue
        idx = np.where(labels == lbl)[0]
        if len(idx) == 0:
            continue
        cent = emb_norm[idx].mean(axis=0, keepdims=True)
        d = cosine_distances(emb_norm[idx], cent).ravel()
        best = idx[np.argsort(d)[:top_m]]
        reps[lbl] = [texts[i] for i in best.tolist()]
    return reps

def _ctfidf_keywords(
    clusters,
    top_k=10,
    ngram_range=(2, 4),
    min_df=3,
    max_df=0.35,
    max_features=100_000,
    stop_words=None,
):
    """
    Class-based TF-IDF over clusters:
    - Build a "document" per cluster by concatenating cluster docs
    - Use CountVectorizer to get phrase counts
    - Compute TF per cluster and IDF across clusters (df = clusters containing term)
    Returns {cluster_label: [keyword phrases]}
    """
    labels = sorted(clusters.keys())
    docs_per_cluster = [" ".join(clusters[lbl]) for lbl in labels]

    vectorizer = CountVectorizer(
        stop_words=stop_words,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    X = vectorizer.fit_transform(docs_per_cluster)  # (C x V) counts

    # TF: normalize counts per cluster
    row_sums = np.asarray(X.sum(axis=1)).ravel() + 1e-12
    tf = X.multiply(1 / row_sums[:, None])

    # IDF across clusters: how many clusters contain the term
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    idf = np.log((len(labels) + 1) / (df + 1)) + 1.0

    ctfidf = tf.multiply(idf)

    terms = np.array(vectorizer.get_feature_names_out())
    out = {}
    for i, lbl in enumerate(labels):
        row = np.asarray(ctfidf[i].todense()).ravel()
        top_idx = row.argsort()[::-1][:top_k]
        out[lbl] = terms[top_idx].tolist()
    return out

def _merge_clusters_by_centroid_similarity(
    texts, emb_norm, labels, target_k, sim_threshold=0.90
):
    """
    Optional: Overcluster -> merge by centroid cosine similarity until you reach target_k,
    or until no mergeable pair above sim_threshold remains.
    """
    unique = sorted([l for l in set(labels) if l != -1])
    if len(unique) <= target_k:
        return labels

    # build cluster -> indices
    cluster_to_idx = {l: np.where(labels == l)[0].tolist() for l in unique}

    def compute_centroids(cluster_to_idx):
        labs = sorted(cluster_to_idx.keys())
        cents = []
        for l in labs:
            idx = cluster_to_idx[l]
            cents.append(emb_norm[idx].mean(axis=0))
        return labs, np.vstack(cents)

    while len(cluster_to_idx) > target_k:
        labs, cents = compute_centroids(cluster_to_idx)
        sims = cosine_similarity(cents)

        np.fill_diagonal(sims, -np.inf)
        i, j = np.unravel_index(np.argmax(sims), sims.shape)
        best_sim = sims[i, j]
        if best_sim < sim_threshold:
            break

        a, b = labs[i], labs[j]
        # merge b into a
        cluster_to_idx[a].extend(cluster_to_idx[b])
        del cluster_to_idx[b]

    # reassign labels to 0..K-1
    new_map = {old: new for new, old in enumerate(sorted(cluster_to_idx.keys()))}
    new_labels = np.full_like(labels, fill_value=-1)
    for old_lbl, idxs in cluster_to_idx.items():
        new_labels[idxs] = new_map[old_lbl]
    return new_labels


def cluster_topic(topic_url: str, n: int = 20000, representative_abstracts: int = 20, random_state: int = 42) \
        -> tuple[dict[int, list[str]], dict[int, int]]:
    """
    Cluster scientific abstracts for an OpenAlex topic and return representative abstracts per cluster.
    :param topic_url: OpenAlex topic URL.
    :param n: Maximum number of papers to load from the topic.
    :param representative_abstracts: Number of representative texts to return per cluster.
    :param random_state: Random seed for reproducible UMAP projection.
    :return: (representatives, cluster_sizes)
        - representatives: dict[int, list[str]] mapping cluster_id -> representative texts
        - cluster_sizes: dict[int, int] mapping cluster_id -> number of texts in cluster
    """
    out_root: str = os.path.join("out", "topics")

    # -----------------------------
    # Constants (kept internal for simplicity)
    # -----------------------------
    embedding_model_name: str = "allenai/specter2_base"
    batch_size: int = 32

    hdb_min_cluster_size: int = 100
    hdb_selection_method: str = "leaf"
    umap_n_components: int = 5
    umap_n_neighbors: int = 20
    umap_min_dist: float = 0.0
    hdb_min_samples: int = 10

    # -----------------------------
    # Load texts
    # -----------------------------
    topic_id: str = topic_url.rstrip("/").split("/")[-1]
    save_dir: str = os.path.join(out_root, topic_id)
    os.makedirs(save_dir, exist_ok=True)

    texts: list[str] = load_topic_title_abstract(topic_url, n=n)
    if not texts:
        print("No papers loaded.")
        return {}, {}

    texts = [_normalize_text(t) for t in texts]
    print(f"Loaded {len(texts)} papers. Computing embeddings ({embedding_model_name})...")

    # -----------------------------
    # Embeddings (CPU/GPU auto)
    # -----------------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(embedding_model_name, device=device)

    emb: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    emb = normalize(emb)  # cosine geometry
    print("Embeddings computed. Shape:", emb.shape)

    # -----------------------------
    # UMAP (for clustering)
    # -----------------------------
    emb_umap: np.ndarray = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        min_dist=umap_min_dist,
        metric="cosine",
        random_state=random_state,
    ).fit_transform(emb)

    # -----------------------------
    # HDBSCAN clustering
    # -----------------------------
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdb_min_cluster_size,
        min_samples=hdb_min_samples,
        metric="euclidean",
        cluster_selection_method=hdb_selection_method,
    )
    labels: np.ndarray = clusterer.fit_predict(emb_umap)

    n_clusters: int = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise: int = int(np.sum(labels == -1))
    print(f"HDBSCAN clusters: {n_clusters} (noise docs: {n_noise})")

    # -----------------------------
    # Pick representative abstracts per cluster
    # -----------------------------
    representatives: dict[int, list[str]] = {}
    cluster_sizes: dict[int, int] = {}

    for c in sorted(set(labels)):
        if c == -1:
            continue  # skip noise by design

        idxs: np.ndarray = np.where(labels == c)[0]
        cluster_sizes[int(c)] = int(len(idxs))

        k: int = min(representative_abstracts, len(idxs))

        cluster_emb: np.ndarray = emb[idxs]
        centroid: np.ndarray = cluster_emb.mean(axis=0, keepdims=True)

        sims: np.ndarray = cosine_similarity(cluster_emb, centroid).ravel()
        top_local: np.ndarray = np.argsort(-sims)[:k]
        top_idxs: np.ndarray = idxs[top_local]

        representatives[int(c)] = [texts[i] for i in top_idxs]

    # -----------------------------
    # Save representatives for TopicGPT step
    # -----------------------------
    reps_path: str = os.path.join(save_dir, f"representatives_top{representative_abstracts}.txt")
    with open(reps_path, "w", encoding="utf-8") as f:
        for c in sorted(representatives.keys()):
            f.write(f"Cluster {c} (n={cluster_sizes[c]}):\n")
            for i, t in enumerate(representatives[c], 1):
                f.write(f"  [{i}] {t}\n")
            f.write("\n")
    print("Saved representatives to:", reps_path)

    return representatives, cluster_sizes


def run_topic_gpt(
    topic_url: str,
    model: str = "meta-llama/Llama-3.3-70B-Instruct",
    temperature: float = 0.2,
    representative_abstracts: int = 20,
    out_root: str = os.path.join("out", "topics"),
) -> dict[int, dict[str, Any]]:
    """
    Run TopicGPT labeling on clustered representative abstracts.
    Loads umbrella topic metadata (display_name + description) from data/openalex/topics.json.
    If the file is missing, calls get_openalex_topics() to create it.

    :param topic_url: OpenAlex topic URL
    :param model: LLM model identifier
    :param temperature: Sampling temperature
    :param representative_abstracts: Number of reps per cluster (must match filename)
    :param out_root: Output root directory
    :return: Dict mapping cluster_id -> TopicGPT result
    """
    # -----------------------------
    # Resolve topic_id and paths
    # -----------------------------
    topic_url = topic_url.rstrip("/")  # CHANGED: normalize for matching
    topic_id: str = topic_url.split("/")[-1]  # e.g. "T10346"
    save_dir: str = os.path.join(out_root, topic_id)

    reps_path: str = os.path.join(save_dir, f"representatives_top{representative_abstracts}.txt")
    if not os.path.exists(reps_path):
        raise FileNotFoundError(f"Missing representatives file: {reps_path}")

    # -----------------------------
    # Load OpenAlex umbrella topic metadata
    # -----------------------------
    topics_path: str = os.path.join("data", "openalex", "topics.json")

    if not os.path.exists(topics_path):
        # Create it, then try again (local import avoids circular import issues)
        from openalex import get_openalex_topics
        get_openalex_topics()
        if not os.path.exists(topics_path):
            raise FileNotFoundError(
                f"{topics_path} not found even after calling get_openalex_topics()."
            )

    with open(topics_path, "r", encoding="utf-8") as f:
        topics_data = json.load(f)

    # topics.json may be a list[dict] or dict[str, dict] depending on your pipeline.
    topic_obj: dict[str, Any] | None = None

    if isinstance(topics_data, list):
        # Each element has e.g. {"id": "https://openalex.org/T10346", ...}
        for t in topics_data:
            if not isinstance(t, dict):
                continue
            tid = str(t.get("id", "")).rstrip("/")
            # CHANGED: also allow matching by plain topic_id if stored that way
            if tid == topic_url or tid.endswith(f"/{topic_id}") or tid == topic_id:
                topic_obj = t
                break

    elif isinstance(topics_data, dict):
        # Could be keyed by topic_id or topic_url
        topic_obj = topics_data.get(topic_id) or topics_data.get(topic_url)

        # Or could be {"results": [...]} depending on your fetch format
        if topic_obj is None and "results" in topics_data and isinstance(topics_data["results"], list):
            for t in topics_data["results"]:
                if not isinstance(t, dict):
                    continue
                tid = str(t.get("id", "")).rstrip("/")
                if tid == topic_url or tid.endswith(f"/{topic_id}") or tid == topic_id:
                    topic_obj = t
                    break

    if topic_obj is None:
        raise KeyError(
            f"Topic {topic_url} ({topic_id}) not found in {topics_path}. "
            f"Check how topics.json is structured."
        )

    umbrella_display_name: str = str(topic_obj.get("display_name", "")).strip()
    umbrella_description: str = str(topic_obj.get("description", "")).strip()

    # Umbrella metadata is REQUIRED for your prompt design
    if not umbrella_display_name or not umbrella_description:
        raise ValueError(
            f"Umbrella metadata missing for {topic_url}.\n"
            f"display_name='{umbrella_display_name}'\n"
            f"description length={len(umbrella_description)}\n"
            f"Fix topics.json or refresh via get_openalex_topics()."
        )

    # -----------------------------
    # Parse representatives file
    # -----------------------------
    clusters: dict[int, list[str]] = {}
    current_cluster: int | None = None

    header_re = re.compile(r"^Cluster\s+(\d+)\s+\(n=\d+\):")
    rep_re = re.compile(r"^\s*\[\d+\]\s+(.*)$")

    with open(reps_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            header = header_re.match(line)
            if header:
                current_cluster = int(header.group(1))
                clusters[current_cluster] = []
                continue

            rep = rep_re.match(line)
            if rep and current_cluster is not None:
                clusters[current_cluster].append(rep.group(1))

    if not clusters:
        raise ValueError("No clusters found in representatives file.")

    # -----------------------------
    # Run TopicGPT
    # -----------------------------
    topic_gpt = TopicGPT(model=model, temperature=temperature)

    results: dict[int, dict[str, Any]] = {}
    for cluster_id in sorted(clusters):
        results[cluster_id] = topic_gpt.label_cluster(
            cluster_id=cluster_id,
            abstracts=clusters[cluster_id],
            umbrella_display_name=umbrella_display_name,
            umbrella_description=umbrella_description,
        )

    # -----------------------------
    # Save results
    # -----------------------------
    out_path = os.path.join(save_dir, "topicgpt_labels.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "topic_id": topic_id,
                "topic_url": topic_url,
                "umbrella_display_name": umbrella_display_name,
                "umbrella_description": umbrella_description,
                "model": model,
                "temperature": temperature,
                "representative_abstracts": representative_abstracts,
                "clusters": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Saved TopicGPT labels to:", out_path)
    return results
