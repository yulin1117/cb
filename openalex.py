import requests
import csv

API_BASE = "https://api.openalex.org/concepts/"
# You may prefer to use “topics” endpoint if fields are there: “https://api.openalex.org/topics”
# But we’ll attempt via concepts and filter by level or domain/parent relationships.

def fetch_concept(concept_id, mailto="youremail@example.com"):
    resp = requests.get(f"{API_BASE}{concept_id}?mailto={mailto}")
    resp.raise_for_status()
    return resp.json()

def fetch_fields(level=0, mailto="youremail@example.com"):
    # Get all concepts at that level
    results = []
    url = f"{API_BASE}?filter=level:{level}&per-page=200&mailto={mailto}"
    while url:
        print("Fetching:", url)
        resp = requests.get(url)
        resp.raise_for_status()
        j = resp.json()
        results.extend(j["results"])
        url = j.get("meta", {}).get("next_page_url")
    return results

def main():
    # Try retrieving level=0 as fields
    fields = fetch_concepts = fetch_fields(level=0)
    print(f"Retrieved {len(fields)} concepts at level=0")
    # Write to CSV
    with open("openalex_fields.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "display_name", "level", "works_count"])
        for c in fields:
            writer.writerow([c["id"], c["display_name"], c.get("level"), c.get("works_count")])
    print("Written to openalex_fields.csv")

if __name__ == "__main__":
    main()
