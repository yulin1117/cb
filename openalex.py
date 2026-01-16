import os
import re
import json
import requests
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sentence_transformers import SentenceTransformer
import hdbscan
from typing import Any

from llm.topic_gpt import TopicGPT


def get_openalex_topics(cache_file="data/openalex/topics.json"):
    """
    Fetch all topics from OpenAlex API or read from cache if available.

    Parameters:
        cache_file (str): Path to JSON file for caching topics.

    Returns:
        list: List of topic dicts.
    """
    # Check if cache exists
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            print(f"Loading topics: {cache_file}")
            return json.load(f)

    print("Topics not found. Fetching topics from OpenAlex API...")

    base_url = "https://api.openalex.org/topics"
    per_page = 200
    topics = []
    cursor = "*"

    while True:
        url = f"{base_url}?per_page={per_page}&cursor={cursor}"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"OpenAlex API request failed: {response.text}")

        data = response.json()
        results = data.get("results", [])
        topics.extend(results)

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    # Save to JSON
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)

    print(f"Topics saved to {cache_file}")

    return topics


def get_works_for_topic(topic_url, n=10000):
    """
    Retrieve n random works for a given OpenAlex topic using primary_topic.id
    and reservoir sampling. First tries to load from disk:
      data/openalex/{topic_id}_works_{n}.json
    Only calls the API if the file does not exist.
    """
    # Extract topic ID ("https://openalex.org/T1234" → "T1234")
    topic_id = topic_url.rstrip("/").split("/")[-1]

    # Prepare directory & file paths
    save_dir = os.path.join("data", "openalex")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{topic_id}_works_{n}.json")

    # Load from disk (if file exists)
    if os.path.exists(save_path):
        print(f"Loading works from disk: {save_path}")
        with open(save_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # API reservoir sampling
    base_url = "https://api.openalex.org/works"
    per_page = 200
    cursor = "*"  # initial cursor for pagination
    reservoir = []
    total_seen = 0

    print(f"Sampling {n} random works for topic {topic_id} via API...")

    while True:
        url = f"{base_url}?filter=primary_topic.id:{topic_id}&per_page={per_page}&cursor={cursor}"
        response = requests.get(url)

        if response.status_code != 200:
            raise Exception(f"OpenAlex API request failed: {response.text}")

        data = response.json()
        results = data.get("results", [])

        if not results:
            break

        for work in results:
            total_seen += 1

            if len(reservoir) < n:
                reservoir.append(work)
            else:
                # Reservoir sampling: randomly replace existing items
                s = random.randint(1, total_seen)
                if s <= n:
                    reservoir[s - 1] = work

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break  # end of pagination

    if not reservoir:
        print(f"No works found for topic {topic_id}")
        return []

    print(f"Processed {total_seen} works; sample size: {len(reservoir)}")

    # Save results as JSON
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(reservoir, f, indent=2, ensure_ascii=False)

    print(f"Saved results to: {save_path}")

    return reservoir


def load_topic_title_abstract(topic_url, n=20000):
    """
    Loads works via get_works_for_topic, keeps only allowed types, English-only,
    rebuilds abstract safely, removes papers with missing title/abstract,
    and returns combined 'Title. Abstract' strings.
    """

    ALLOWED_TYPES = {
        "article",
        "book-chapter",
        "preprint",
        "dissertation",
        "book",
        "review",
        "report",
    }

    def reconstruct_abstract(abstract_inverted_index):
        """Safely rebuild abstract from OpenAlex abstract_inverted_index."""
        if not abstract_inverted_index or not isinstance(abstract_inverted_index, dict):
            return None

        clean_items = []
        for word, positions in abstract_inverted_index.items():
            if not isinstance(word, str):
                continue
            if not isinstance(positions, list):
                continue

            positions = [p for p in positions if isinstance(p, int) and p >= 0]
            if not positions:
                continue

            clean_items.append((word, positions))

        if not clean_items:
            return None

        max_pos = max(pos for _, positions in clean_items for pos in positions)
        words = [""] * (max_pos + 1)

        for word, positions in clean_items:
            for pos in positions:
                if 0 <= pos < len(words):
                    words[pos] = word

        text = " ".join(w for w in words if w)
        return text if text.strip() else None

    raw_works = get_works_for_topic(topic_url=topic_url, n=n)
    cleaned_papers = []

    for w in raw_works:

        # Filter by allowed type
        if w.get("type") not in ALLOWED_TYPES:
            continue

        # English only
        if w.get("language") != "en":
            continue

        title = w.get("title")
        inv = w.get("abstract_inverted_index")
        if not title or not inv:
            continue

        abstract = reconstruct_abstract(inv)
        if not abstract:
            continue

        combined = f"{title}. {abstract}"

        # Length filter
        if len(combined) < 100:
            continue

        cleaned_papers.append(combined)

    return cleaned_papers


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

    :param topic_url: OpenAlex topic URL
    :param model: LLM model identifier
    :param temperature: Sampling temperature
    :param representative_abstracts: Number of reps per cluster (must match filename)
    :param out_root: Output root directory
    :return: Dict mapping cluster_id -> TopicGPT result
    """
    topic_id: str = topic_url.rstrip("/").split("/")[-1]
    save_dir: str = os.path.join(out_root, topic_id)

    reps_path = os.path.join(
        save_dir, f"representatives_top{representative_abstracts}.txt"
    )
    if not os.path.exists(reps_path):
        raise FileNotFoundError(f"Missing representatives file: {reps_path}")

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
                "model": model,
                "temperature": temperature,
                "clusters": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Saved TopicGPT labels to:", out_path)
    return results