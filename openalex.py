import os
import json
import requests
import random


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
