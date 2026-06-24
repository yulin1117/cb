import os
import json
import requests
import random
import time
import csv
import hashlib
from collections import Counter
from nltk.tokenize import sent_tokenize
import fasttext
from sentence_transformers import SentenceTransformer, util
def get_openalex_topics(cache_file="../data/openalex/topics.json"):
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

def _get_fasttext_model():
    _ft_model = None
    FASTTEXT_MODEL_PATH = "lid.176.ftz"
    _ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
    return _ft_model


def _get_sentence_transformer_model():
    _st_model = None
    ST_MODEL_NAME = "all-MiniLM-L6-v2"
    _st_model = SentenceTransformer(ST_MODEL_NAME)
    return _st_model


def lang_detect(text, model, threshold=0.2):
    """
    Sentence-by-sentence validation using FastText. Filters out documents containing more non-eng sentences than threshold.
    """
    sentences = sent_tokenize(text)
    valid_sentences_count=0
    non_eng_sentences  = 0  
    for s in sentences:
        s = s.strip().replace("\n", " ")
        if len(s) < 5:
            continue
        valid_sentences_count += 1
        labels, probs = model.predict(s, k=1)
        lang = labels[0].replace("__label__", "")

        if lang != "en" :
            non_eng_sentences += 1
    if valid_sentences_count > 0:
        if non_eng_sentences / valid_sentences_count >= threshold:
                return False        
    return True
def duplicate_detect(abstract: str, seen_hashes: set[str]) -> bool:
    """
    Deduplication Check: Returns True if abstract duplicate collision is detected.
    """
    abstract_key = hashlib.md5(abstract.strip().lower().encode()).hexdigest()
    if abstract_key in seen_hashes:
        return True
    seen_hashes.add(abstract_key)
    return False
def content_detect(title: str, abstract: str, st_model) -> bool:
    """
    Compute the similarity of the title and the abstract. Returns True if it matches semantic expectations (>= 0.50).
    """
    # Execute semantic alignment validation
    if st_model is not None:
        try:
            t_emb = st_model.encode(title.strip())
            a_emb = st_model.encode(abstract.strip())
            similarity_score = util.cos_sim(t_emb, a_emb).item()
            if similarity_score < 0.5:
                return False
        except Exception:
            return False  # Fail-safe reject on vectorization error
    return True


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


def get_works_for_topic(topic_url: str, n: int = 5000, random_state: int = 42,
                        chunk_size: int = 2000, max_rounds: int = 10):
    """
    Retrieve works for a given OpenAlex topic using primary_topic.id and OpenAlex server-side sampling (sample + seed),
    and "top up" sampling until we have n COMPLETE papers according to the SAME screening criteria used in
    load_topic_title_abstract().

    This function intentionally mirrors the downstream screening rules:
      - allowed types only
      - English only
      - non-empty title
      - abstract_inverted_index present
      - abstract can be reconstructed (non-empty)
      - combined length >= 500

    Results are cached to disk under:
      data/openalex/{topic_id}_works.json

    Only calls the API if the file does not exist.

    :param topic_url: OpenAlex topic URL (e.g., https://openalex.org/T1234)
    :param n: Number of complete works to retrieve (default 5000; OpenAlex sample supports up to 10000 per round)
    :param random_state: Base seed for deterministic sampling (seed increments each round)
    :param chunk_size: Sample size per round (default 2000)
    :param max_rounds: Maximum number of sampling rounds
    :return: List of work dicts that pass the screening criteria (length == n unless insufficient data)
    """
    topic_id = topic_url.rstrip("/").split("/")[-1]

    save_dir = os.path.join("../data", "openalex")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{topic_id}_works.json")

    # Load from disk (if file exists)
    if os.path.exists(save_path):
        print(f"Loading works from disk: {save_path}")
        with open(save_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"Calling OpenAlex API for topic {topic_id} (target n={n}, base_seed={random_state})")

    # -----------------------------
    # Screening criteria (mirrors load_topic_title_abstract)
    # -----------------------------
    ALLOWED_TYPES = {
        "article",
        "book-chapter",
        "preprint",
        "dissertation",
        "book",
        "review",
        "report",
    }
    ft_model=_get_fasttext_model()
    st_model=_get_sentence_transformer_model()
    def is_complete_work(w: dict) -> bool:
        """Apply the same screening as load_topic_title_abstract() would."""
        if w.get("type") not in ALLOWED_TYPES:
            return False
        if w.get("language") != "en":
            return False

        title = w.get("title")
        inv = w.get("abstract_inverted_index")
        if not title or not inv:
            return False

        abstract = reconstruct_abstract(inv)
        if not abstract:
            return False

        combined = f"{title}. {abstract}"
        if len(combined) < 500:
            return False

        return True

    # -----------------------------
    # OpenAlex API config
    # -----------------------------
    base_url = "https://api.openalex.org/works"
    per_page = 200

    headers = {
        "User-Agent": "CitationBiasBenchmark (mailto:tobias.schreieder@tu-dresden.de)"
    }

    # Push the cheap filters into the API call (does not change criteria; just reduces waste)
    type_filter = "|".join(sorted(ALLOWED_TYPES))
    select_fields = "id,title,type,language,abstract_inverted_index"

    complete_works: list[dict] = []
    seen_ids: set[str] = set()
    seen_abstract_hashes: set[str] = set()
    # -----------------------------
    # Top-up sampling loop
    # -----------------------------
    for round_idx in range(max_rounds):
        if len(complete_works) >= n:
            break

        remaining = n - len(complete_works)
        this_sample = min(10000, max(int(chunk_size), int(remaining)))
        seed = int(random_state) + round_idx

        print(f"  Round {round_idx + 1}/{max_rounds}: request sample={this_sample}, seed={seed}")

        max_pages = (this_sample + per_page - 1) // per_page
        added_this_round = 0

        for page in range(1, max_pages + 1):
            url = (
                f"{base_url}"
                f"?filter=primary_topic.id:{topic_id},has_abstract:true,language:en,type:{type_filter}"
                f"&sample={this_sample}"
                f"&seed={seed}"
                f"&select={select_fields}"
                f"&per-page={per_page}"
                f"&page={page}"
            )

            # Backoff for transient errors / rate limits
            for attempt in range(6):
                resp = requests.get(url, headers=headers, timeout=60)
                if resp.status_code == 200:
                    break
                if resp.status_code in (429, 500, 502, 503, 504):
                    sleep_s = min(60, 2 ** attempt) + random.random()
                    time.sleep(sleep_s)
                    continue
                raise Exception(f"OpenAlex API request failed ({resp.status_code}): {resp.text}")
            else:
                raise Exception(f"OpenAlex API request failed after retries: {url}")

            data = resp.json()
            results = data.get("results", [])
            if not results:
                break

            for w in results:
                wid = w.get("id")
                if not isinstance(wid, str) or not wid:
                    continue
                if wid in seen_ids:
                    continue
                seen_ids.add(wid)
                if not isinstance(w, dict):
                    continue

                # Reconstruct abstract content from API payload
                title = w.get("title") or ""
                inv = w.get("abstract_inverted_index")
                abstract = reconstruct_abstract(inv) or ""

                # ─── rule-based filters ───

                # 1. Missing Metadata filter check
                if not is_complete_work(w):
                    continue

                # 2. Duplicate Abstract check
                if duplicate_detect(abstract, seen_abstract_hashes):
                    continue

                combined_text = f"{title}. {abstract}"
                # 3. English language check (FastText sentence-by-sentence)
                if not lang_detect(combined_text, ft_model):
                    continue

                # 4. Content match check (SentenceTransformer check)
                if not content_detect(title, abstract, st_model):
                    continue
                
                w["reconstructed_abstract"] = abstract
                complete_works.append(w)
                added_this_round += 1

                if len(complete_works) >= n:
                    break

            if len(complete_works) >= n:
                break

        print(f"    added_complete={added_this_round}, total_complete={len(complete_works)}")

        if added_this_round == 0:
            print("    No new complete works added this round; stopping early.")
            break

    complete_works = complete_works[:n]

    if not complete_works:
        print(f"No complete works found for topic {topic_id}")
        return []

    # Save to disk
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(complete_works, f, ensure_ascii=False)

    if len(complete_works) < n:
        print(f"Saved {len(complete_works)} complete works (requested {n}) to: {save_path}")
    else:
        print(f"Saved {len(complete_works)} complete works to: {save_path}")

    return complete_works


def get_works_for_topic_reservoir_sampling(topic_url, n=10000):
    """
    OLD METHOD
    Retrieve n random works for a given OpenAlex topic using primary_topic.id
    and reservoir sampling. First tries to load from disk:
      data/openalex/{topic_id}_works_{n}.json
    Only calls the API if the file does not exist.
    """
    # Extract topic ID ("https://openalex.org/T1234" → "T1234")
    topic_id = topic_url.rstrip("/").split("/")[-1]

    # Prepare directory & file paths
    save_dir = os.path.join("../data", "openalex")
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


def load_topic_title_abstract(topic_url):
    """
    Loads works via get_works_for_topic, rebuilds abstracts safely, and returns
    a list of dictionaries with OpenAlex ID and combined text.

    Filtering by type/language/title/abstract/length is intentionally NOT done here,
    because get_works_for_topic() already enforces the same screening criteria.

    NOTE: If texts appear lowercased later, this is caused by _normalize_text()
    in cluster_topic(), not by this loader.
    """

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

    raw_works = get_works_for_topic(topic_url=topic_url)

    papers: list[dict[str, str]] = []

    for w in raw_works:
        if not isinstance(w, dict):
            continue

        work_id = w.get("id")
        title = w.get("title")
        inv = w.get("abstract_inverted_index")

        if not work_id or not title or not inv:
            continue

        abstract = reconstruct_abstract(inv)
        if not abstract:
            continue

        combined = f"{title}. {abstract}"

        papers.append({
            "id": str(work_id),
            "title": title,
            "abstract": abstract,
            "text": combined
        })

    return papers

