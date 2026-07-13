import os
import re
import csv
import json
import time
import random
import requests
import numpy as np
import pandas as pd

# Safely import SentenceTransformers for vector evaluations
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

def reconstruct_abstract(abstract_inverted_index) -> str | None:
    """
    Safely reconstructs plain text from OpenAlex abstract_inverted_index.
    """
    if not abstract_inverted_index or not isinstance(abstract_inverted_index, dict):
        return None

    clean_items = []
    for word, positions in abstract_inverted_index.items():
        if not isinstance(word, str) or not isinstance(positions, list):
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
    return text.strip() if text.strip() else None

def is_paywall_garbage(text: str) -> tuple[bool, str]:
    """
    Applies heuristic checkers to detect publisher metadata, paywall redirection prompts, 
    and general web-scraping noise common in non-OA articles.
    """
    t = text.lower()

    # 1. JATS/HTML Tag Contaminations
    jats_html = [
        "<jats:p>", "<jats:sec>", "<jats:title>", "<jats:italic>",
        "<jats:bold>", "</jats:", "jats:p>", "<div", "<b>", "</b>", 
        "<i>", "</i>", "<p>", "</p>", "<br", "&lt;jats:", "&lt;div"
    ]
    for tag in jats_html:
        if tag in t:
            return True, f"JATS/HTML Tag Found ('{tag}')"

    # 2. Paywall & Citation Widget Triggers
    paywall_keywords = {
        "this content is only available as a pdf": "PDF Paywall Indicator",
        "you do not currently have access": "Access Denied Message",
        "log in to your account": "Login Prompt Redirection",
        "purchase this article": "Purchase Paywall Prompt",
        "download citation file:": "Citation Exporter Widget",
        "ris (zotero)": "Zotero Citation Reference block",
        "reference manager": "Reference Manager Reference block",
        "easybib": "EasyBib Widget",
        "refworks": "Refworks Reference block",
        "share on facebook": "Social Widget Residual",
        "share on twitter": "Social Widget Residual"
    }
    for kw, label in paywall_keywords.items():
        if kw in t:
            return True, label

    # 3. Structural Regex Residual Patterns (DOIs, page counts, or ISSN headers)
    if re.search(r"^.{0,150}https?://doi\.org/10\.\d{4,}", text):
        return True, "Early DOI Redirection Link"
    if re.search(r"^(volume|vol\.?)\s+\d+,\s+(issue|iss\.?)\s+\d+", t):
        return True, "Volume/Issue Header Residual"
    if re.search(r"(online issn|print issn|issn[:\s])\s*\d{4}-\d{3}[\dx]", t):
        return True, "ISSN Header Noise"
    if re.search(r"^©\s+\d{4}\s+[a-z][a-za-z\s]{3,60}$", t):
        return True, "Pure Copyright Line Abstract"

    return False, "Passed"

def fetch_non_oa_works(topic_id: str = "T13674", num_papers: int = 150) -> list[dict]:
    """
    Queries OpenAlex for papers marked strictly as non-open-access (is_oa=false)
    that possess abstract indexing, targeted within a benchmark topic.
    """
    print(f"\n📡 Requesting non-OA records for Topic [{topic_id}] (Target count: {num_papers})...")
    base_url = "https://api.openalex.org/works"
    results = []
    page = 1
    page_size = 50
    
    headers = {
        "User-Agent": "CitationBiasBenchmark/GarbageScanner (mailto:tobias.schreieder@tu-dresden.de)"
    }
    
    while len(results) < num_papers:
        # Request works matching non-OA filters
        ALLOWED_TYPES = {
        "article",
        "book-chapter",
        "preprint",
        "dissertation",
        "book",
        "review",
        "report",
        }
        type_filter = "|".join(sorted(ALLOWED_TYPES))
        url = (
            f"{base_url}?filter=primary_topic.id:{topic_id},open_access.is_oa:false,has_abstract:true,language:en,type:{type_filter}"
            f"&per-page={page_size}&page={page}"
        )
        
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                page_results = data.get("results", [])
                if not page_results:
                    break
                results.extend(page_results)
                print(f"   Fetched page {page}... Current pool: {len(results)} papers.")
                page += 1
                time.sleep(0.15)  # Respect API limits
            else:
                print(f"❌ API Request failed with status code: {resp.status_code}")
                break
        except Exception as e:
            print(f"❌ Connection error: {e}")
            break
            
    return results[:num_papers]

def run_scanner(topic_id: str = "T13674", paper_count: int = 150, threshold: float = 0.50):
    """
    Main orchestration loop: Downloads non-OA corpus, extracts fields, 
    applies physical filters, measures semantic vector checks, and writes reports.
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("❌ Error: sentence-transformers is missing! Please install it using:")
        print("   pip install sentence-transformers")
        return

    # 1. Gather Candidate Dataset
    raw_works = fetch_non_oa_works(topic_id, paper_count)
    if not raw_works:
        print("❌ Failed to retrieve works from OpenAlex API.")
        return

    print("\n🤖 Loading local SentenceTransformer ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    scanned_records = []
    
    # Counter Metrics
    total_scanned = len(raw_works)
    heuristic_failed_count = 0
    semantic_failed_count = 0
    total_garbage_count = 0

    print(f"\n🔍 Processing scan checks for {total_scanned} non-OA abstracts...")

    for idx, w in enumerate(raw_works, 1):
        work_id = w.get("id", "").split("/")[-1] or "Unknown"
        title = w.get("title") or ""
        inv_index = w.get("abstract_inverted_index")
        
        # Abstract Reconstruction
        abstract_text = reconstruct_abstract(inv_index) if inv_index else ""
        
        if not title or not abstract_text:
            # Mark as empty garbage
            scanned_records.append({
                "id": work_id, "title": title, "abstract_len": 0,
                "heuristic_status": "FAILED", "heuristic_reason": "Empty Title or Abstract",
                "similarity_score": 0.0, "classification": "GARBAGE"
            })
            heuristic_failed_count += 1
            total_garbage_count += 1
            continue

        # Check Heuristics
        is_h_garbage, h_reason = is_paywall_garbage(abstract_text)
        
        # Calculate Vector Similarity
        try:
            t_emb = model.encode(title.strip())
            a_emb = model.encode(abstract_text.strip())
            sim_score = float(util.cos_sim(t_emb, a_emb).item())
        except Exception:
            sim_score = 0.0

        # Classify and log
        is_semantic_garbage = sim_score < threshold
        
        if is_h_garbage:
            classification = "GARBAGE"
            heuristic_failed_count += 1
            total_garbage_count += 1
        elif is_semantic_garbage:
            classification = "GARBAGE"
            semantic_failed_count += 1
            total_garbage_count += 1
        else:
            classification = "CLEAN"

        scanned_records.append({
            "id": work_id,
            "title": title,
            "abstract": abstract_text,
            "heuristic_status": "FAILED" if is_h_garbage else "PASSED",
            "heuristic_reason": h_reason,
            "similarity_score": round(sim_score, 4),
            "classification": classification
        })

    df = pd.DataFrame(scanned_records)
    csv_file = "non_oa_garbage_scan.csv"
    df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    print(f"\n💾 Scanned results successfully written to: {os.path.abspath(csv_file)}")

    garbage_rate = (total_garbage_count / total_scanned) * 100
    h_failure_rate = (heuristic_failed_count / total_scanned) * 100
    s_failure_rate = (semantic_failed_count / total_scanned) * 100

    md_file = "non_oa_garbage_report.md"
    md_content = f"""# Diagnostic Scan: Non-OA Paywall & Semantic Garbage Audit
This analytical scanner parses scientific works marked as **strictly non-open-access (`open_access.is_oa = false`)** on OpenAlex to audit structural paywall noise and title-abstract mismatches.

---

## 📊 Core Diagnostic Metrics
- **Target Topic**: `{topic_id}`
- **Total Non-OA Works Audited**: {total_scanned} papers
- **Semantic Similarity Threshold**: `{threshold}`
- **Total Garbage Content Identified**: **{total_garbage_count} papers**
- **🎯 Overall Garbage Content Ratio**: **{garbage_rate:.2f}%**

---

## 🔍 Structural Failure Vectors
| Failure Category | Triggered Count | Percentage | Operational Threat |
| :--- | :---: | :---: | :--- |
| **Heuristic Garbage** (Paywalls, Socials, Widgets) | {heuristic_failed_count} | {h_failure_rate:.2f}% | Corrupts clustering and poisons downstream LLM/TopicGPT labels. |
| **Semantic Mismatch** (Similarity < {threshold}) | {semantic_failed_count} | {s_failure_rate:.2f}% | Distorts topic boundary consistency. |

---

## 💡 Heuristic Diagnostic Conclusions
### Why Non-OA Papers Suffer High Garbage Rates
Under the `is_oa = false` restriction, publisher sites frequently block OpenAlex web crawlers. Consequently, rather than a genuine abstract, the `abstract_inverted_index` fields are loaded with:
1. **Redirection Residuals** (e.g., *"This content is only available as a PDF"*, *"You do not currently have access"*).
2. **Citation Widget Captures** (e.g., *"Download citation file: Ris (Zotero) Reference Manager EasyBib"*).
3. **Journal Metadata Overflow** (Page counters, online ISSNs, editor headers).

---
*Report dynamically generated via non_oa_garbage_scanner.py.*
"""
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"💾 Executive summary report successfully saved to: {os.path.abspath(md_file)}")

    print("\n" + "="*80)
    print("                 🏁 SCANNER METRICS SUMMARY")
    print("="*80)
    print(f" ● Total Non-OA Papers Audited         : {total_scanned} works")
    print(f" ● Triggered Heuristic Garbage Filter   : {heuristic_failed_count} papers ({h_failure_rate:.2f}%)")
    print(f" ● Triggered Semantic Mismatch Filter  : {semantic_failed_count} papers ({s_failure_rate:.2f}%)")
    print("-" * 80)
    print(f" 🎯 OVERALL GARBAGE CONTENT RATIO     : {garbage_rate:.2f}%")
    print("="*80 + "\n")

if __name__ == "__main__":
    # You can change the topic_id to test different scientific scopes
    run_scanner(topic_id="T13674", paper_count=1000, threshold=0.50)
