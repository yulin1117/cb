import os
import random
import re
import json
import csv
from collections import defaultdict

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
from openalex import load_topic_title_abstract, get_openalex_topics, get_works_for_topic

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


def _cluster_scores_from_embeddings(emb: np.ndarray, labels: np.ndarray) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """
    Compute coherence, distinctiveness, and score per cluster using ORIGINAL embedding space.
    Assumes emb is L2-normalized (cosine geometry).
    """
    cluster_ids = [int(c) for c in sorted(set(labels)) if c != -1]
    if not cluster_ids:
        return {}, {}, {}

    # ---- centroids ----
    centroids = {}
    members = {}
    for c in cluster_ids:
        idxs = np.where(labels == c)[0]
        members[c] = idxs
        mu = emb[idxs].mean(axis=0, keepdims=True)
        mu /= (np.linalg.norm(mu) + 1e-12)  # normalize centroid
        centroids[c] = mu

    # stack centroids for fast pairwise cosine
    C = np.vstack([centroids[c] for c in cluster_ids])  # shape: (k, d)
    CC = C @ C.T  # cosine similarities since rows normalized

    # ---- coherence ----
    coherence = {}
    for i, c in enumerate(cluster_ids):
        idxs = members[c]
        mu = C[i:i+1]  # (1, d)
        # mean cosine to centroid (dot product since normalized)
        coherence[c] = float((emb[idxs] @ mu.T).mean())

    # ---- distinctiveness ----
    distinctiveness = {}
    for i, c in enumerate(cluster_ids):
        row = CC[i].copy()
        row[i] = -np.inf  # ignore self
        max_sim = float(np.max(row)) if len(cluster_ids) > 1 else 0.0
        distinctiveness[c] = float(1.0 - max_sim)

    # ---- combined score ----
    score = {c: coherence[c] * distinctiveness[c] for c in cluster_ids}
    return coherence, distinctiveness, score


def select_openalex_topics(n: int = 25, random_state: int = 42) -> dict[str, list[str]]:
    """
    Select OpenAlex topic URLs stratified by OpenAlex field.

    Loads all available OpenAlex topics via get_openalex_topics() and groups them by
    topic["field"]["display_name"]. For each field, selects up to n topic ids uniformly
    at random using the provided random_state.

    The selected topics are persisted to disk under
    data/openalex/selected_topics.json for reproducibility. If the file already exists
    with matching parameters (n_per_field and random_state), the stored selection is
    loaded and returned. Otherwise, the file is overwritten.

    :param n: Number of topics to sample per field (upper bound; fields with fewer topics return all).
    :param random_state: Random seed for deterministic sampling.
    :return: Dict mapping field_display_name -> list of topic ids (OpenAlex URLs).
    """
    out_dir = os.path.join("data", "openalex")
    out_path = os.path.join(out_dir, "selected_topics.json")

    # -----------------------------
    # Fast path: load existing selection if parameters match
    # -----------------------------
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if (
                isinstance(existing, dict)
                and existing.get("n_per_field") == int(n)
                and existing.get("random_state") == int(random_state)
                and isinstance(existing.get("fields"), dict)
            ):
                return existing["fields"]
        except Exception:
            # Fall through to recomputation
            pass

    # -----------------------------
    # Load all OpenAlex topics
    # -----------------------------
    oa_topics: list[dict[str, Any]] = get_openalex_topics()
    if not isinstance(oa_topics, list) or not oa_topics:
        return {}

    topics_by_field: dict[str, list[str]] = defaultdict(list)

    for t in oa_topics:
        if not isinstance(t, dict):
            continue

        topic_id = t.get("id")
        field = t.get("field")

        if not isinstance(topic_id, str) or not topic_id.strip():
            continue
        if not isinstance(field, dict):
            continue

        field_name = field.get("display_name")
        if not isinstance(field_name, str) or not field_name.strip():
            continue

        topics_by_field[field_name.strip()].append(topic_id.strip())

    # -----------------------------
    # Deterministic sampling per field
    # -----------------------------
    rng = random.Random(random_state)

    selected: dict[str, list[str]] = {}
    for field_name in sorted(topics_by_field.keys()):
        ids = topics_by_field[field_name]
        if not ids:
            continue

        ids = list(ids)
        rng.shuffle(ids)
        selected[field_name] = ids[: min(int(n), len(ids))]

    # -----------------------------
    # Persist (overwrite) selection
    # -----------------------------
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_per_field": int(n),
                "random_state": int(random_state),
                "fields": selected,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Saved selected OpenAlex topics to:", out_path)

    return selected


def cluster_topic(
        topic_url: str,
        representative_abstracts: int = 20,
        random_state: int = 42,
        return_all_clusters: bool = False,
        size_cap: int = 2000,
        distinct_top_k: int = 1,
        top_k_candidates: int = 10,
) -> tuple[dict[int, list[dict[str, str]]], dict[int, int], dict[int, dict[str, float]]]:
    """
    Cluster scientific abstracts for an OpenAlex topic and return representative abstracts.

    The method embeds abstracts using a scientific sentence embedding model, clusters them
    with UMAP + HDBSCAN, and scores clusters for semantic coherence and distinctiveness in the
    original embedding space. Unlike an early-selection pipeline, this method does not need to
    commit to a single cluster before downstream labeling: it can return either (a) all clusters
    or (b) a top-K set of candidate clusters ranked by the unsupervised score, enabling final
    selection after TopicGPT (e.g., using an LLM confidence threshold).

    Representative abstracts are chosen from each returned cluster using MMR (Maximal Marginal
    Relevance) diversity sampling on a centroid-relevance candidate pool. This reduces redundancy
    among representatives while keeping them highly on-topic.

    :param topic_url: OpenAlex topic URL identifying the umbrella topic to process.
    :param representative_abstracts: Number of representative abstracts to return per cluster.
    :param random_state: Random seed for reproducible UMAP projection and clustering.
    :param return_all_clusters: Whether to return representative abstracts for all clusters.
        If False, returns only the top-K candidate clusters by unsupervised score.
    :param size_cap: Upper bound used when computing the cluster size prior to prevent very
        large clusters from dominating the ranking.
    :param distinct_top_k: Number of most similar sibling cluster centroids considered when
        computing cluster distinctiveness.
    :param top_k_candidates: Number of highest-scoring clusters to return when
        return_all_clusters is False.
    :return: (representatives, cluster_sizes, cluster_metrics)
        - representatives: dict[int, list[dict]] mapping cluster_id to representative paper dicts
          with at least {"id": ..., "text": ...}.
        - cluster_sizes: dict[int, int] mapping cluster_id to the number of papers in each returned cluster.
        - cluster_metrics: dict[int, dict[str, float]] mapping cluster_id to coherence/distinctiveness/score.
    """
    out_root: str = os.path.join("out", "topics")

    # -----------------------------
    # Constants (kept internal for simplicity)
    # -----------------------------
    embedding_model_name: str = "allenai/specter2_base"
    batch_size: int = 32

    hdb_min_cluster_size: int = 50
    hdb_selection_method: str = "leaf"
    umap_n_components: int = 10
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.0
    hdb_min_samples: int = 5

    # MMR selection hyperparameters
    mmr_lambda: float = 0.7
    mmr_pool_min: int = 50
    mmr_pool_mult: int = 5

    # -----------------------------
    # Helpers
    # -----------------------------
    def _compute_cluster_scores(
            emb_norm: np.ndarray,
            labels: np.ndarray,
            sizes: dict[int, int],
            *,
            size_cap_: int,
            distinct_top_k_: int
    ) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
        """
        Compute coherence, distinctiveness, and final score for each non-noise cluster
        using ORIGINAL embedding space (emb_norm must be L2-normalized).
        """
        cluster_ids = [int(c) for c in sorted(set(labels)) if c != -1]
        if not cluster_ids:
            return {}, {}, {}

        # centroids (normalized)
        centroids = []
        member_idxs = {}
        for c in cluster_ids:
            idxs = np.where(labels == c)[0]
            member_idxs[c] = idxs
            mu = emb_norm[idxs].mean(axis=0)
            mu = mu / (np.linalg.norm(mu) + 1e-12)
            centroids.append(mu)
        C = np.vstack(centroids)  # (k, d), rows normalized

        # pairwise centroid cosine
        CC = C @ C.T  # (k, k)

        # coherence: mean cosine to centroid
        coherence = {}
        for i, c in enumerate(cluster_ids):
            idxs = member_idxs[c]
            mu = C[i]  # (d,)
            coherence[c] = float((emb_norm[idxs] @ mu).mean())

        # distinctiveness: 1 - agg(top sims to other centroids)
        distinctiveness = {}
        k = len(cluster_ids)
        for i, c in enumerate(cluster_ids):
            sims = CC[i].copy()
            sims[i] = -np.inf
            if k == 1:
                agg = 0.0
            else:
                dtk = max(1, int(distinct_top_k_))
                dtk = min(dtk, k - 1)
                top = np.partition(sims, -dtk)[-dtk:]
                agg = float(top.mean()) if dtk > 1 else float(top.max())
            distinctiveness[c] = float(1.0 - agg)

        # size prior: sqrt(min(n_c, cap)/cap)
        score = {}
        cap = max(1, int(size_cap_))
        for c in cluster_ids:
            n_c = int(sizes.get(c, 0))
            size_prior = float(np.sqrt(min(n_c, cap) / cap))
            score[c] = float(coherence[c] * distinctiveness[c] * size_prior)

        return coherence, distinctiveness, score

    def _mmr_select(
            cand_emb: np.ndarray,
            cand_rel: np.ndarray,
            k: int,
            lam: float = 0.7,
    ) -> np.ndarray:
        """
        Maximal Marginal Relevance selection on a candidate pool.

        :param cand_emb: (L, d) L2-normalized embeddings (cosine = dot)
        :param cand_rel: (L,) relevance scores (e.g., cosine to centroid)
        :param k: number to select
        :param lam: relevance vs diversity tradeoff (0..1); higher = more relevance
        :return: indices into cand_emb of selected items (length k)
        """
        L = int(cand_emb.shape[0])
        if L == 0:
            return np.array([], dtype=int)

        k = min(int(k), L)
        cand_rel = np.asarray(cand_rel).ravel()

        # Similarity among candidates (cosine, since normalized)
        sim_mat = cand_emb @ cand_emb.T  # (L, L)

        selected: list[int] = []
        first = int(np.argmax(cand_rel))
        selected.append(first)

        remaining = set(range(L))
        remaining.remove(first)

        while len(selected) < k and remaining:
            best_i = None
            best_score = -1e18

            for i in remaining:
                redundancy = float(np.max(sim_mat[i, selected])) if selected else 0.0
                mmr = float(lam * cand_rel[i] - (1.0 - lam) * redundancy)
                if mmr > best_score:
                    best_score = mmr
                    best_i = i

            if best_i is None:
                break

            selected.append(int(best_i))
            remaining.remove(int(best_i))

        return np.array(selected, dtype=int)

    # -----------------------------
    # Load papers (id + text)
    # -----------------------------
    topic_id: str = topic_url.rstrip("/").split("/")[-1]
    save_dir: str = os.path.join(out_root, topic_id)
    os.makedirs(save_dir, exist_ok=True)

    papers: list[dict[str, str]] = load_topic_title_abstract(topic_url)
    if not papers:
        print("No papers loaded.")
        return {}, {}, {}

    # Extract ids and texts; keep original dicts aligned by index
    paper_ids: list[str] = [str(p.get("id", "")) for p in papers]
    texts_raw: list[str] = [str(p.get("text", "")) for p in papers]

    # Defensive: drop empty id/text rows (should be rare if upstream screening is correct)
    keep = [i for i, (pid, txt) in enumerate(zip(paper_ids, texts_raw)) if pid and txt]
    if len(keep) != len(papers):
        papers = [papers[i] for i in keep]
        paper_ids = [paper_ids[i] for i in keep]
        texts_raw = [texts_raw[i] for i in keep]

    if not papers:
        print("No usable papers after id/text validation.")
        return {}, {}, {}

    # NOTE: lowercasing, if any, happens here via _normalize_text
    texts: list[str] = [_normalize_text(t) for t in texts_raw]

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

    # normalize for cosine geometry
    emb = normalize(emb)
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
    # Cluster sizes (all clusters, used for scoring)
    # -----------------------------
    all_cluster_sizes: dict[int, int] = {}
    for c in sorted(set(labels)):
        if c == -1:
            continue
        idxs = np.where(labels == c)[0]
        all_cluster_sizes[int(c)] = int(len(idxs))

    # -----------------------------
    # Score clusters (ORIGINAL embedding space)
    # -----------------------------
    coherence, distinctiveness, score = _compute_cluster_scores(
        emb, labels, all_cluster_sizes,
        size_cap_=size_cap,
        distinct_top_k_=distinct_top_k,
    )

    if not score:
        print("No non-noise clusters to score.")
        return {}, {}, {}

    # Persist scores for auditing/debugging
    scores_path: str = os.path.join(save_dir, "cluster_scores.tsv")
    with open(scores_path, "w", encoding="utf-8") as f:
        f.write("cluster_id\tn\tcoherence\tdistinctiveness\tscore\n")
        for c in sorted(score.keys()):
            f.write(
                f"{c}\t{all_cluster_sizes[c]}\t{coherence[c]:.6f}\t{distinctiveness[c]:.6f}\t{score[c]:.6f}\n"
            )
    print("Saved cluster scores to:", scores_path)

    # -----------------------------
    # Decide which clusters to return (no final selection here)
    # -----------------------------
    if return_all_clusters:
        clusters_to_emit = sorted(score.keys())
        print(f"Returning all scored clusters: {len(clusters_to_emit)}")
    else:
        k = max(1, int(top_k_candidates))
        clusters_sorted = sorted(score.keys(), key=lambda c: score[c], reverse=True)
        clusters_to_emit = clusters_sorted[:min(k, len(clusters_sorted))]
        print(f"Returning top-{len(clusters_to_emit)} candidate clusters by score "
              f"(top_k_candidates={k}).")

    # -----------------------------
    # Pick representative papers per returned cluster (MMR)
    # -----------------------------
    representatives: dict[int, list[dict[str, str]]] = {}
    cluster_sizes: dict[int, int] = {}
    cluster_metrics: dict[int, dict[str, float]] = {}

    for c in clusters_to_emit:
        idxs: np.ndarray = np.where(labels == c)[0]
        cluster_sizes[int(c)] = int(len(idxs))

        k_rep: int = min(representative_abstracts, len(idxs))
        if k_rep <= 0:
            continue

        cluster_emb: np.ndarray = emb[idxs]  # normalized
        centroid: np.ndarray = cluster_emb.mean(axis=0, keepdims=True)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        sims: np.ndarray = (cluster_emb @ centroid.T).ravel()  # relevance to centroid

        # Candidate pool: top-L by centroid similarity, then MMR downselect to k_rep
        L = min(len(idxs), max(mmr_pool_min, mmr_pool_mult * k_rep))
        cand_local = np.argsort(-sims)[:L]           # indices into cluster_emb (local)
        cand_emb = cluster_emb[cand_local]           # (L, d)
        cand_rel = sims[cand_local]                  # (L,)

        mmr_local = _mmr_select(cand_emb, cand_rel, k_rep, lam=mmr_lambda)

        # Map back to global indices into `papers`
        top_global_idxs: np.ndarray = idxs[cand_local[mmr_local]]

        # Store representative paper dicts (id + raw/original fields)
        reps_list: list[dict[str, str]] = []
        for gi in top_global_idxs:
            p = papers[int(gi)]
            reps_list.append({
                "id": str(p.get("id", "")),
                "text": str(p.get("text", "")),
            })

        representatives[int(c)] = reps_list
        cluster_metrics[int(c)] = {
            "coherence": float(coherence[c]),
            "distinctiveness": float(distinctiveness[c]),
            "score": float(score[c]),
        }

    # -----------------------------
    # Save representatives (JSON only) for TopicGPT step
    # -----------------------------
    reps_json_path: str = os.path.join(save_dir, f"representatives_top{representative_abstracts}.json")
    with open(reps_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "topic_id": topic_id,
                "topic_url": topic_url.rstrip("/"),
                "representative_abstracts": representative_abstracts,
                "clusters": {
                    str(c): {
                        "n": cluster_sizes.get(c),
                        "metrics": cluster_metrics.get(c, {}),
                        "papers": representatives.get(c, []),
                    }
                    for c in clusters_to_emit
                    if c in representatives
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("Saved representatives JSON to:", reps_json_path)

    return representatives, cluster_sizes, cluster_metrics


def run_topic_gpt(topic_url: str, model: str = "meta-llama/Llama-3.3-70B-Instruct", temperature: float = 0.2,
                  representative_abstracts: int = 20, out_root: str = os.path.join("out", "topics")) \
        -> dict[int, dict[str, Any]]:
    """
    Run TopicGPT labeling on clustered representative abstracts.
    Loads umbrella topic metadata (display_name + description) from data/openalex/topics.json.
    If the file is missing, calls get_openalex_topics() to create it.

    :param topic_url: OpenAlex topic URL
    :param model: LLM model identifier
    :param temperature: Sampling temperature
    :param representative_abstracts: Number of reps per cluster (must match filename)
    :param out_root: Output root directory
    :return: Dict mapping cluster_id -> TopicGPT result (includes cluster n)
    """
    # -----------------------------
    # Resolve topic_id and paths
    # -----------------------------
    topic_url = topic_url.rstrip("/")
    topic_id: str = topic_url.split("/")[-1]
    save_dir: str = os.path.join(out_root, topic_id)

    # UPDATED: load JSON representatives instead of TXT
    reps_path: str = os.path.join(save_dir, f"representatives_top{representative_abstracts}.json")
    if not os.path.exists(reps_path):
        raise FileNotFoundError(f"Missing representatives file: {reps_path}")

    # -----------------------------
    # Load OpenAlex umbrella topic metadata
    # -----------------------------
    topics_path: str = os.path.join("data", "openalex", "topics.json")

    if not os.path.exists(topics_path):
        from openalex import get_openalex_topics  # local import avoids circular imports
        get_openalex_topics()
        if not os.path.exists(topics_path):
            raise FileNotFoundError(
                f"{topics_path} not found even after calling get_openalex_topics()."
            )

    with open(topics_path, "r", encoding="utf-8") as f:
        topics_data = json.load(f)

    topic_obj: dict[str, Any] | None = None

    if isinstance(topics_data, list):
        for t in topics_data:
            if not isinstance(t, dict):
                continue
            tid = str(t.get("id", "")).rstrip("/")
            if tid == topic_url or tid.endswith(f"/{topic_id}") or tid == topic_id:
                topic_obj = t
                break

    elif isinstance(topics_data, dict):
        topic_obj = topics_data.get(topic_id) or topics_data.get(topic_url)

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

    if not umbrella_display_name or not umbrella_description:
        raise ValueError(
            f"Umbrella metadata missing for {topic_url}.\n"
            f"display_name='{umbrella_display_name}'\n"
            f"description length={len(umbrella_description)}\n"
            f"Fix topics.json or refresh via get_openalex_topics()."
        )

    # -----------------------------
    # Load cluster metrics (coh/dist/score) from cluster_scores.tsv (if present)
    # -----------------------------
    metrics_by_cluster: dict[int, dict[str, float]] = {}
    scores_path: str = os.path.join(save_dir, "cluster_scores.tsv")
    if os.path.exists(scores_path):
        with open(scores_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            try:
                cid_i = header.index("cluster_id")
                coh_i = header.index("coherence")
                dist_i = header.index("distinctiveness")
                score_i = header.index("score")
            except ValueError as e:
                raise ValueError(
                    f"cluster_scores.tsv must contain columns: cluster_id, coherence, distinctiveness, score. Got: {header}"
                ) from e

            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                try:
                    cid = int(parts[cid_i])
                    metrics_by_cluster[cid] = {
                        "coherence": float(parts[coh_i]),
                        "distinctiveness": float(parts[dist_i]),
                        "score": float(parts[score_i]),
                    }
                except Exception:
                    continue

    # -----------------------------
    # Load representatives from JSON
    # -----------------------------
    with open(reps_path, "r", encoding="utf-8") as f:
        reps_data = json.load(f)

    clusters_obj = reps_data.get("clusters", {})
    if not isinstance(clusters_obj, dict) or not clusters_obj:
        raise ValueError("No clusters found in representatives JSON file.")

    clusters: dict[int, list[str]] = {}
    cluster_n: dict[int, int] = {}

    for cid_str, cdata in clusters_obj.items():
        try:
            cid = int(cid_str)
        except Exception:
            continue

        if not isinstance(cdata, dict):
            continue

        n_val = cdata.get("n")
        try:
            cluster_n[cid] = int(n_val) if n_val is not None else None
        except Exception:
            cluster_n[cid] = None

        papers = cdata.get("papers", [])
        if not isinstance(papers, list) or not papers:
            continue

        # TopicGPT expects a list[str]; keep the exact same textual payload as before.
        abstracts_for_llm: list[str] = []
        for p in papers:
            if not isinstance(p, dict):
                continue
            txt = str(p.get("text", "")).strip()
            if txt:
                abstracts_for_llm.append(txt)

        if abstracts_for_llm:
            clusters[cid] = abstracts_for_llm

    if not clusters:
        raise ValueError("No usable clusters/papers found in representatives JSON file.")

    # -----------------------------
    # Run TopicGPT
    # -----------------------------
    topic_gpt = TopicGPT(model=model, temperature=temperature)

    results: dict[int, dict[str, Any]] = {}
    for cluster_id in sorted(clusters):
        label = topic_gpt.label_cluster(
            cluster_id=cluster_id,
            abstracts=clusters[cluster_id],
            umbrella_display_name=umbrella_display_name,
            umbrella_description=umbrella_description,
        )
        # Only include n in the result (cluster size from JSON)
        label["n"] = cluster_n.get(cluster_id)

        # attach unsupervised metrics if available
        if cluster_id in metrics_by_cluster:
            label.update(metrics_by_cluster[cluster_id])

        results[cluster_id] = label

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



def select_topic_from_topicgpt(topic_url: str) -> tuple[int, dict]:
    """
    Select the final cluster/topic after TopicGPT labeling.

    This function is intentionally file-driven: it resolves the topic-specific output
    directory from the OpenAlex topic URL, then reads
      - cluster scores from "cluster_scores.tsv" (produced by cluster_topic), and
      - TopicGPT outputs from "topicgpt_labels.json" (produced by run_topic_gpt),
    and selects the highest-scoring cluster among TopicGPT-labeled clusters whose
    confidence is exactly "high".

    If multiple clusters receive an identical TopicGPT topic_name (after normalization),
    only the highest-scoring cluster per topic_name is kept for selection (deduplication
    is applied within the "high" candidate pool).

    Confidence thresholding is not parameterized here by design: only "high" is accepted.
    If no "high" candidate exists, it falls back to the highest-scoring cluster that has
    any TopicGPT output, and if that also fails, to the highest-scoring cluster overall.

    :param topic_url: OpenAlex topic URL identifying the umbrella topic to process.
    :return: (selected_cluster_id, selected_topicgpt_json)
        - selected_cluster_id: Cluster id selected as final fine-grained topic.
        - selected_topicgpt_json: TopicGPT label payload for the selected cluster (may be {} on fallback).
    """
    representative_abstracts: int = 20
    out_root: str = os.path.join("out", "topics")

    # -----------------------------
    # Resolve topic_id and paths (aligned with run_topic_gpt)
    # -----------------------------
    topic_url = topic_url.rstrip("/")
    topic_id: str = topic_url.split("/")[-1]
    save_dir: str = os.path.join(out_root, topic_id)

    # cluster scores
    scores_path: str = os.path.join(save_dir, "cluster_scores.tsv")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"Missing cluster scores file: {scores_path}")

    # TopicGPT output
    labels_path: str = os.path.join(save_dir, "topicgpt_labels.json")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing TopicGPT labels file: {labels_path}")

    # Optional: ensure reps file exists for consistency/debugging (not strictly required for selection)
    reps_path: str = os.path.join(save_dir, f"representatives_top{representative_abstracts}.txt")
    if not os.path.exists(reps_path):
        pass

    # -----------------------------
    # Load cluster scores TSV
    # -----------------------------
    scores: dict[int, float] = {}
    with open(scores_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        try:
            cid_i = header.index("cluster_id")
            score_i = header.index("score")
        except ValueError as e:
            raise ValueError(
                f"cluster_scores.tsv must contain columns: cluster_id, score. Got: {header}"
            ) from e

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            try:
                c = int(parts[cid_i])
                s = float(parts[score_i])
            except Exception:
                continue
            scores[c] = s

    if not scores:
        raise ValueError(f"No cluster scores found in: {scores_path}")

    # -----------------------------
    # Load TopicGPT outputs (run_topic_gpt format)
    # -----------------------------
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clusters_obj = data.get("clusters")
    if not isinstance(clusters_obj, dict) or not clusters_obj:
        raise ValueError(f"No 'clusters' found in TopicGPT labels file: {labels_path}")

    topic_by_cluster: dict[int, dict] = {}
    for k, v in clusters_obj.items():
        try:
            cid = int(k)
        except Exception:
            if isinstance(k, int):
                cid = k
            else:
                continue
        if isinstance(v, dict):
            topic_by_cluster[cid] = v

    if not topic_by_cluster:
        raise ValueError(f"No usable cluster payloads found in: {labels_path}")

    # -----------------------------
    # Selection: confidence == "high" only (with topic_name dedup)
    # -----------------------------
    high_candidates = [
        c for c, payload in topic_by_cluster.items()
        if isinstance(payload, dict) and str(payload.get("confidence", "")).lower() == "high"
    ]

    def _best_by_score(cands: list[int]) -> int | None:
        cands_scored = [c for c in cands if c in scores]
        if not cands_scored:
            return None
        return max(cands_scored, key=lambda c: scores[c])

    def _norm_topic_name(x) -> str:
        # Normalize for exact-string collisions: lowercase, strip, collapse whitespace
        return " ".join(str(x).strip().lower().split())

    # Keep only the highest-scoring cluster per normalized topic_name among high candidates
    best_cluster_per_name: dict[str, int] = {}
    for c in high_candidates:
        if c not in scores:
            continue
        payload = topic_by_cluster.get(c, {})
        name = _norm_topic_name(payload.get("topic_name", ""))

        # If no usable name, keep it unique so we don't accidentally collapse unrelated clusters
        if not name:
            name = f"__missing_name__:{c}"

        prev = best_cluster_per_name.get(name)
        if prev is None or scores[c] > scores.get(prev, float("-inf")):
            best_cluster_per_name[name] = c

    high_candidates_deduped = list(best_cluster_per_name.values())

    selected_cluster = _best_by_score(high_candidates_deduped)

    # Fallback 1: any TopicGPT output (regardless of confidence)
    if selected_cluster is None:
        selected_cluster = _best_by_score(list(topic_by_cluster.keys()))

    # Fallback 2: best unsupervised overall (even without TopicGPT output)
    if selected_cluster is None:
        selected_cluster = max(scores.keys(), key=lambda c: scores[c])

    selected_payload = topic_by_cluster.get(selected_cluster, {})
    if not isinstance(selected_payload, dict):
        selected_payload = {}

    return int(selected_cluster), selected_payload


def create_openalex_dataset(n_per_field: int = 20, works_n: int = 5000, random_state: int = 4):
    """
    Create the OpenAlex-based benchmark dataset by running:
      - topic selection (stratified by field),
      - work retrieval (cached),
      - clustering,
      - TopicGPT labeling,
      - final cluster selection (confidence=="high", dedup by identical topic_name, then best score).

    Stores a CSV with columns:
      - Topic (ours)            -> TopicGPT selected topic_name
      - Description (ours)      -> TopicGPT selected description
      - OpenAlex-Topic          -> "<openalex_display_name> (<topic_id>)"
      - Field                  -> "<field_display_name> (<field_id>)"

    :param n_per_field: number of OpenAlex topics sampled per field
    :param works_n: number of complete works to retrieve per topic (get_works_for_topic handles top-up)
    :param random_state: random seed used for topic selection + work sampling
    """
    out_csv: str = os.path.join("data", "openalex", "openalex_dataset.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # -----------------------------
    # Select OpenAlex topics (stratified by field)
    # -----------------------------
    topics_by_field: dict[str, list[str]] = select_openalex_topics(n=n_per_field, random_state=random_state)

    # Flatten topic URLs
    topic_urls: list[str] = []
    for _field_name, topic_ids in topics_by_field.items():
        for tid in topic_ids:
            tid = str(tid).strip()
            if not tid:
                continue
            # allow either full URL or bare id
            if tid.startswith("http"):
                topic_urls.append(tid.rstrip("/"))
            else:
                topic_urls.append(f"https://openalex.org/{tid}".rstrip("/"))

    if not topic_urls:
        print("No topics selected.")
        return

    # -----------------------------
    # Load OpenAlex topic metadata for name + field (id/name)
    # -----------------------------
    oa_topics: list[dict[str, Any]] = get_openalex_topics()
    topic_meta_by_id: dict[str, dict[str, Any]] = {}

    for t in oa_topics:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("id", "")).rstrip("/")
        if not tid:
            continue
        topic_meta_by_id[tid] = t
        # also index by bare id (Txxxxx) for convenience
        topic_meta_by_id[tid.split("/")[-1]] = t

    # -----------------------------
    # Prepare CSV writer
    # -----------------------------
    rows: list[dict[str, str]] = []

    total = len(topic_urls)
    for i, topic_url in enumerate(topic_urls, 1):
        topic_url = topic_url.rstrip("/")
        topic_id = topic_url.split("/")[-1]

        print(f"\n[{i}/{total}] Processing topic {topic_id} ...")

        # Metadata lookup
        meta = topic_meta_by_id.get(topic_url) or topic_meta_by_id.get(topic_id) or {}
        oa_display_name = str(meta.get("display_name", "")).strip() or topic_id

        field_obj = meta.get("field") if isinstance(meta, dict) else None
        field_id = ""
        field_name = ""
        if isinstance(field_obj, dict):
            field_id = str(field_obj.get("id", "")).strip()
            field_name = str(field_obj.get("display_name", "")).strip()

        # Format as requested: name + id
        openalex_topic_str = f"{oa_display_name} ({topic_id})"
        if field_name and field_id:
            field_str = f"{field_name} ({field_id.split('/')[-1]})"
        elif field_name:
            field_str = field_name
        elif field_id:
            field_str = field_id
        else:
            field_str = ""

        # -----------------------------
        # Run pipeline (with safety)
        # -----------------------------
        try:
            get_works_for_topic(topic_url=topic_url, n=works_n, random_state=random_state)
        except Exception as e:
            print(f"  [ERROR] get_works_for_topic failed for {topic_id}: {e}")
            continue

        try:
            cluster_topic(topic_url=topic_url)
        except Exception as e:
            print(f"  [ERROR] cluster_topic failed for {topic_id}: {e}")
            continue

        try:
            _labels = run_topic_gpt(topic_url=topic_url)
        except Exception as e:
            print(f"  [ERROR] run_topic_gpt failed for {topic_id}: {e}")
            continue

        try:
            selected_cluster, selected_payload = select_topic_from_topicgpt(topic_url=topic_url)
        except Exception as e:
            print(f"  [ERROR] select_topic_from_topicgpt failed for {topic_id}: {e}")
            continue

        ours_topic = str(selected_payload.get("topic_name", "")).strip()
        ours_desc = str(selected_payload.get("description", "")).strip()

        if not ours_topic:
            # Keep a traceable fallback rather than silently writing empty
            ours_topic = f"(missing topic_name) cluster={selected_cluster}"

        rows.append({
            "Topic": ours_topic,
            "Description": ours_desc,
            "OpenAlex-Topic": openalex_topic_str,
            "Field": field_str,
        })

        print(f"  Selected cluster {selected_cluster}: {ours_topic}")

    # -----------------------------
    # Write CSV
    # -----------------------------
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Topic", "Description", "OpenAlex-Topic", "Field"],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nSaved dataset CSV with {len(rows)} rows to: {out_csv}")
