import os
import re
import json
import requests
import random
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

from sentence_transformers import SentenceTransformer


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


# -----------------------------
# Main: state-of-the-art pipeline
# -----------------------------
def cluster_topic(
    topic_url,
    n=20000,
    # clustering options
    cluster_method="hdbscan",  # "hdbscan" (recommended), "kmeans", or "kmeans_overcluster_merge"
    num_clusters=30,           # used for kmeans; for overcluster, this is initial K
    target_clusters=10,        # used for overcluster-merge (final target)
    # UMAP for clustering (high-D) + for visualization (2D)
    umap_cluster_n_components=10,
    umap_cluster_n_neighbors=30,
    umap_cluster_min_dist=0.0,
    umap_vis_n_neighbors=15,
    umap_vis_min_dist=0.1,
    # HDBSCAN settings
    hdb_min_cluster_size=75,
    hdb_min_samples=10,
    hdb_selection_method="eom",  # or "leaf" for more granular clusters
    # keywords (c-TF-IDF)
    top_k_keywords=8,
    ctfidf_ngram_range=(2, 4),    # phrases (task-like)
    ctfidf_min_df=3,
    ctfidf_max_df=0.35,
    extra_stopwords=None,
    # reps for LLM / inspection
    representative_per_cluster=5,
    # embedding model
    device="cuda",
    embedding_model_name="allenai/specter2_base",
    batch_size=32,
    # merge threshold (only for overcluster-merge)
    merge_sim_threshold=0.90,
    # output
    out_root=os.path.join("out", "topics"),
    random_state=42,
):
    """
    SOTA-ish scientific abstract topic extraction:
    - SPECTER2 embeddings (normalized)
    - clustering: HDBSCAN on UMAP(10D) (recommended) OR KMeans variants
    - topic representation: c-TF-IDF phrase keywords (2-4 grams) + domain stopwords
    - representative docs per cluster for LLM labeling
    - UMAP 2D visualization

    Returns:
      clusters: {label: [texts]}
      cluster_keywords: {label: [phrases]}
      representatives: {label: [rep_texts]}
    """
    # -----------------------------
    # LOAD DATA
    # -----------------------------
    topic_id = topic_url.rstrip("/").split("/")[-1]
    save_dir = os.path.join(out_root, topic_id)
    os.makedirs(save_dir, exist_ok=True)

    texts = load_topic_title_abstract(topic_url, n=n)
    if not texts:
        print("No papers loaded.")
        return None, None, None

    # Normalize text lightly (helps vectorizer + keywording)
    texts = [_normalize_text(t) for t in texts]

    print(f"Loaded {len(texts)} papers. Computing embeddings ({embedding_model_name})...")

    # -----------------------------
    # EMBEDDINGS (SPECTER2) + L2 normalize
    # -----------------------------
    model = SentenceTransformer(embedding_model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # we'll do it explicitly below
    )
    embeddings = normalize(embeddings)  # L2 normalize -> cosine geometry
    print("Embeddings computed. Shape:", embeddings.shape)

    # -----------------------------
    # CLUSTERING
    # -----------------------------
    cluster_labels = None

    if cluster_method.lower() == "hdbscan":
        try:
            import hdbscan
        except ImportError as e:
            raise ImportError(
                "hdbscan is required for cluster_method='hdbscan'. Install: pip install hdbscan"
            ) from e

        print("UMAP (for clustering) -> HDBSCAN clustering...")
        umap_cluster = umap.UMAP(
            n_neighbors=umap_cluster_n_neighbors,
            n_components=umap_cluster_n_components,
            min_dist=umap_cluster_min_dist,
            metric="cosine",
            random_state=random_state,
        )
        emb_umap = umap_cluster.fit_transform(embeddings)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=hdb_min_cluster_size,
            min_samples=hdb_min_samples,
            metric="euclidean",  # UMAP output is euclidean
            cluster_selection_method=hdb_selection_method,
        )
        cluster_labels = clusterer.fit_predict(emb_umap)

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = int(np.sum(cluster_labels == -1))
        print(f"HDBSCAN clusters: {n_clusters} (noise docs: {n_noise})")

        # If you want to *force* all docs into clusters, you can optionally
        # post-assign noise to nearest centroid here. (Not doing it by default.)

    elif cluster_method.lower() == "kmeans":
        print(f"KMeans clustering (K={num_clusters})...")
        km = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
        cluster_labels = km.fit_predict(embeddings)

    elif cluster_method.lower() == "kmeans_overcluster_merge":
        print(f"KMeans overcluster (K={num_clusters}) then merge to ~{target_clusters}...")
        km = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
        initial_labels = km.fit_predict(embeddings)

        cluster_labels = _merge_clusters_by_centroid_similarity(
            texts=texts,
            emb_norm=embeddings,
            labels=initial_labels,
            target_k=target_clusters,
            sim_threshold=merge_sim_threshold,
        )
        final_k = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"Final clusters after merge: {final_k}")

    else:
        raise ValueError("cluster_method must be one of: 'hdbscan', 'kmeans', 'kmeans_overcluster_merge'")

    # Build cluster -> docs map (skip noise cluster -1)
    clusters = {}
    for text, lbl in zip(texts, cluster_labels):
        if lbl == -1:
            continue
        clusters.setdefault(int(lbl), []).append(text)

    print(f"Total non-noise clusters: {len(clusters)}")

    # -----------------------------
    # REPRESENTATIVE DOCS (for labeling / LLM)
    # -----------------------------
    representatives = _representative_docs(
        texts=texts,
        emb_norm=embeddings,
        labels=cluster_labels,
        top_m=representative_per_cluster,
    )

    reps_path = os.path.join(save_dir, "representatives.txt")
    with open(reps_path, "w", encoding="utf-8") as f:
        for lbl in sorted(representatives.keys()):
            f.write(f"Cluster {lbl} (n={len(clusters.get(lbl, []))}):\n")
            for i, t in enumerate(representatives[lbl], 1):
                f.write(f"  [{i}] {t}\n")
            f.write("\n")
    print("Saved representatives to:", reps_path)

    # -----------------------------
    # KEYWORDS: c-TF-IDF phrases (task-like)
    # -----------------------------
    stop_words = _build_stopwords(extra_stopwords)

    # If you want even more "task-like" phrases, keep (2,4) grams;
    # if you also want acronyms/unigrams (e.g., NER, QA), run an extra unigram pass separately.
    cluster_keywords = _ctfidf_keywords(
        clusters=clusters,
        top_k=top_k_keywords,
        ngram_range=ctfidf_ngram_range,
        min_df=ctfidf_min_df,
        max_df=ctfidf_max_df,
        stop_words=stop_words,
    )

    keywords_path = os.path.join(save_dir, "keywords_ctfidf.txt")
    with open(keywords_path, "w", encoding="utf-8") as f:
        for lbl in sorted(cluster_keywords.keys()):
            f.write(f"Cluster {lbl}: {', '.join(cluster_keywords[lbl])}\n")
    print("Saved c-TF-IDF keywords to:", keywords_path)

    # -----------------------------
    # UMAP 2D VISUALIZATION (for inspection only)
    # -----------------------------
    print("UMAP 2D visualization...")
    reducer = umap.UMAP(
        n_neighbors=umap_vis_n_neighbors,
        min_dist=umap_vis_min_dist,
        metric="cosine",
        random_state=random_state,
    )
    embedding_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    palette = plt.get_cmap("tab20")
    for lbl in sorted(set(cluster_labels)):
        if lbl == -1:
            continue
        idxs = np.where(cluster_labels == lbl)[0]
        plt.scatter(
            embedding_2d[idxs, 0],
            embedding_2d[idxs, 1],
            s=18,
            c=[palette(int(lbl) % 20)],
            label=f"Cluster {lbl}",
            alpha=0.55,
        )

    plt.title(f"Clusters for topic {topic_id} ({cluster_method})")
    plt.legend(fontsize=8, markerscale=1.2, frameon=False)
    plot_path = os.path.join(save_dir, "cluster_umap2d.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved cluster plot to:", plot_path)

    return clusters, cluster_keywords, representatives
