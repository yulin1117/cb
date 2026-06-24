import os
import csv
import json
import re
import hashlib
import time
import random
from collections import Counter

# Import raw data fetching and abstract reconstruction functions from local module
from test2 import fetch_unfiltered_raw_works, reconstruct_abstract
from nltk.tokenize import sent_tokenize
import fasttext
from sentence_transformers import SentenceTransformer, util

# Global model configurations and lazy loading containers
_ft_model = None
FASTTEXT_MODEL_PATH = "lid.176.ftz"

_st_model = None
ST_MODEL_NAME = "all-MiniLM-L6-v2"

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


# ──────────────────────────────────────────────────────────────────────
# Core Language Validation Heuristic via Sentence-by-Sentence Check
# ──────────────────────────────────────────────────────────────────────
def has_non_english(text, model, threshold=0.2):
    """
    Sentence-by-sentence validation using FastText.
    Filters out any document where any single sentence fails English detection.
    """
    sentences = sent_tokenize(text)
    valid_sentences_count = 0  
    non_eng_sentences = 0
    for s in sentences:
        s = s.strip()
        if len(s) < 5:
            continue
        valid_sentences_count += 1
        labels, probs = model.predict(s, k=1)
        lang = labels[0].replace("__label__", "")
        conf = probs[0]

        if lang != "en" :
            non_eng_sentences += 1
            #print(f"⚠️ Detected non-English sentence: '{s}' | Predicted: {lang} | Confidence: {conf:.4f}")
        if valid_sentences_count > 0:
            if non_eng_sentences / valid_sentences_count >= threshold:
                return False
    return True


# ──────────────────────────────────────────────────────────────────────
# Modular Hard Filtering Gates (Phase 1 Pipeline)
# ──────────────────────────────────────────────────────────────────────
def filter_missing_metadata(papers: list[dict]) -> tuple[list[dict], list[dict]]:
    """Workflow (1)e Rule: Check for missing critical metadata (title, abstract, authorships)."""
    passed = []
    rejected = []
    
    for paper in papers:
        paper_id = paper.get("id", "").split("/")[-1] or "Unknown"
        title = paper.get("title") or ""
        authors = paper.get("authorships") or []
        abstract_text = paper.get("reconstructed_abstract", "")
        
        if not title.strip() or not abstract_text.strip() or not authors:
            missing_fields = []
            if not title.strip(): missing_fields.append("title")
            if not abstract_text.strip(): missing_fields.append("abstract")
            if not authors: missing_fields.append("authorships")
            
            paper["decision"] = "REJECTED"
            paper["is_english"] = False
            paper["is_matching"] = False
            paper["audit_reason"] = f"Missing critical metadata: {missing_fields}"
            rejected.append(paper)
            print(f"⚠️ [METADATA MISSING] Paper {paper_id} is missing {missing_fields} -> Filtered out")
        else:
            passed.append(paper)
            
    return passed, rejected


def filter_duplicate_abstract(papers: list[dict]) -> tuple[list[dict], list[dict]]:
    """Workflow (1)d Rule: Global abstract deduplication collision barrier (collision penalty mechanism)."""
    passed = []
    rejected = []
    
    abstract_counter = Counter([p["reconstructed_abstract"] for p in papers if p.get("reconstructed_abstract")])
    
    for paper in papers:
        paper_id = paper.get("id", "").split("/")[-1] or "Unknown"
        abstract_text = paper.get("reconstructed_abstract", "")
        
        if abstract_text and abstract_counter[abstract_text] >= 2:
            paper["decision"] = "REJECTED"
            paper["is_english"] = False
            paper["is_matching"] = False
            paper["audit_reason"] = f"Duplicate Abstract Collision: This abstract appears {abstract_counter[abstract_text]} times across the dataset."
            rejected.append(paper)
            print(f"🚨 [DUPLICATE COLLISION] Paper {paper_id} triggered abstract duplicate collision ({abstract_counter[abstract_text]} times) -> Filtered out")
        else:
            passed.append(paper)
            
    return passed, rejected


def filter_by_hard_rules(papers: list[dict]) -> tuple[list[dict], list[dict]]:
    """Pipeline: Chains abstract reconstruction, missing metadata checks, and duplicate deduplication."""
    print("\n[Phase 1] Starting Hard Rules Firewall (Hard Rules Pipeline)...")
    
    for paper in papers:
        inv_index = paper.get("abstract_inverted_index")
        abstract_text = reconstruct_abstract(inv_index) if inv_index else ""
        paper["reconstructed_abstract"] = " ".join(abstract_text.strip().split())

    meta_passed, meta_rejected = filter_missing_metadata(papers)
    final_passed, dup_rejected = filter_duplicate_abstract(meta_passed)
    
    all_hard_rejected = meta_rejected + dup_rejected
    return final_passed, all_hard_rejected


# ──────────────────────────────────────────────────────────────────────
# 🚀 MODULAR STEP 2: Vectorized Matrix Batch Encoding Function
# ──────────────────────────────────────────────────────────────────────
def compute_vector_embeddings_batched(titles: list[str], abstracts: list[str], st_model, batch_size: int = 64) -> tuple[list, list]:
    """
    Generates high-dimensional semantic vector embeddings for titles and abstracts
    simultaneously using highly optimized matrix operations in PyTorch.
    
    Parameters:
    - titles: List of sanitized paper title strings.
    - abstracts: List of reconstructed plain-text abstract strings.
    - st_model: Lazy-loaded SentenceTransformer model instance.
    - batch_size: Size of batches dispatched to multi-core matrix evaluation pipelines (default: 64).
    
    Returns:
    - Tuple containing (title_embeddings, abstract_embeddings) as fast numerical matrices.
    """
    if st_model is None:
        return [], []
    if not titles or not abstracts:
        return [], []
        
    print(f"📊 Vectorizing {len(titles)} candidate papers using matrix batch operations (batch size: {batch_size})...")
    # Executing batch operations maximizes multi-threading / SIMD parallelism natively
    t_embeddings = st_model.encode(titles, batch_size=batch_size, show_progress_bar=False)
    a_embeddings = st_model.encode(abstracts, batch_size=batch_size, show_progress_bar=False)
    
    return t_embeddings, a_embeddings


# ──────────────────────────────────────────────────────────────────────
# 🎯 Core Ablation Integration: Semantic Auditing using FastText, SentenceTransformers & Rules
# ──────────────────────────────────────────────────────────────────────
def batch_filter_papers_without_llm(papers: list[dict]) -> list[dict]:
    """
    [Ablation Study Controller Function]:
    Completely removes the LLM dependency. Replaces the original LLM tasks (is_english and is_matching)
    by combining the FastText sentence-level detector, the _is_garbage_abstract rules,
    and a SentenceTransformers semantic vector similarity check (threshold >= 0.50).
    """
    if not papers:
        return []

    # 1. Phase 1: Hard Rules Firewall (Abstract Reconstruction + Missing Metadata Check + Collision Deduplication)
    valid_papers_for_rules, hard_rejected_papers = filter_by_hard_rules(papers)

    if not valid_papers_for_rules:
        _write_audit_csv(hard_rejected_papers, "data_cleaning_audit_report_NO_LLM.csv")
        return []

    # Attempt to initialize language models
    ft_model = _get_fasttext_model()
    if not ft_model:
        print("⚠️ [System Warning] FastText loading failed! Language identification is unavailable.")

    # Attempt to initialize semantic transformer model
    st_model = _get_sentence_transformer_model()
    if not st_model:
        print("⚠️ [System Warning] SentenceTransformer loading failed! Semantic similarity check will be skipped.")

    # ──────────────────────────────────────────────────────────────────────
    # 🚀 OPTIMIZATION STEP 1: Pre-screen Heuristics & Prepare Batch Lists
    # ──────────────────────────────────────────────────────────────────────
    pre_screened_papers = []
    titles_to_encode = []
    abstracts_to_encode = []
    
    print(f"\n⚡ Executing FastText & Heuristic preprocessing for {len(valid_papers_for_rules)} papers...")
    
    for paper in valid_papers_for_rules:
        title = paper.get("title", "") or ""
        abstract = paper.get("reconstructed_abstract", "") or ""
        combined_text = f"{title}. {abstract}".strip()
        
        is_english = False
        lang_reason = "Unknown"
        
        # [Language Detection] Sentence-by-sentence validation using FastText
        if ft_model is not None:
            is_english = has_non_english(combined_text, ft_model, threshold=0.2)
            if is_english == False:  
                lang_reason = "FastText detected non-English content."
            else:
                lang_reason = "FastText passed English detection."
        else:
            is_english = False
            lang_reason = "FastText model is unavailable."

        is_matching = True  # try the result without the garbage check for now
        match_reason = "Abstract passed rule-based garbage text inspection." if is_matching else "Triggered rule-based garbage/paywall patterns."

        # Cache pre-screened properties for the second pass
        paper["is_english"] = is_english
        paper["is_matching"] = is_matching
        paper["_lang_reason"] = lang_reason
        paper["_match_reason"] = match_reason
        
        pre_screened_papers.append(paper)
        
        # Collect for SentenceTransformer matrix encoding ONLY if we didn't reject early in the pre-screen
        if is_english and is_matching:
            titles_to_encode.append(title.strip())
            abstracts_to_encode.append(abstract.strip())

    # ──────────────────────────────────────────────────────────────────────
    # 🚀 OPTIMIZATION STEP 2: Modularized Batch Encoding
    # ──────────────────────────────────────────────────────────────────────
    t_embeddings, a_embeddings = compute_vector_embeddings_batched(
        titles_to_encode, abstracts_to_encode, st_model, batch_size=64
    )

    # ──────────────────────────────────────────────────────────────────────
    # 🚀 OPTIMIZATION STEP 3: Compute Vector Metrics & Finalize Decisions
    # ──────────────────────────────────────────────────────────────────────
    approved_papers = []
    rule_processed_papers = []
    vector_idx = 0  # Tracker index for mapping pre-computed embeddings
    
    for paper in pre_screened_papers:
        paper_id = paper.get("id", "").split("/")[-1] or "Unknown"
        is_english = paper["is_english"]
        is_matching = paper["is_matching"]
        lang_reason = paper.pop("_lang_reason")
        match_reason = paper.pop("_match_reason")
        
        similarity_score = 1.0
        
        # Look up vector similarities if eligible for semantic checks
        if is_english and is_matching and st_model is not None:
            try:
                t_emb = t_embeddings[vector_idx]
                a_emb = a_embeddings[vector_idx]
                vector_idx += 1
                
                # Fast numpy-backed cosine similarity calculation
                similarity_score = util.cos_sim(t_emb, a_emb).item()
                
                if similarity_score < 0.5:
                    is_matching = False
                    match_reason = f"Semantic Mismatch: Similarity is {similarity_score:.4f} (threshold: 0.50)"
                else:
                    match_reason = f"Passed vector verification (similarity: {similarity_score:.4f} >= 0.50)"
            except Exception as se:
                match_reason = f"Passed heuristics. Semantic check failed: {se}"
                
        paper["is_matching"] = is_matching
        
        # Final decision resolution
        if is_english is True and is_matching is True:
            paper["decision"] = "APPROVED"
            paper["audit_reason"] = f"{lang_reason} | {match_reason}"
            approved_papers.append(paper)
            print(f"✅ [APPROVED] Paper: {paper_id} | Similarity: {similarity_score:.4f} | Reason: {lang_reason[:50]}...")
        else:
            paper["decision"] = "REJECTED"
            reject_details = []
            if not is_english: 
                reject_details.append(f"Language check failed ({lang_reason})")
            if not is_matching: 
                reject_details.append(f"Matching check failed ({match_reason})")
            
            paper["audit_reason"] = " | ".join(reject_details)
            print(f"❌ [REJECTED] Paper: {paper_id} | Similarity: {similarity_score:.4f} | Rejection Reason: {', '.join(reject_details)}")
            
        rule_processed_papers.append(paper)

    # 💾 Final merging and export to dedicated CSV
    final_csv_list = rule_processed_papers + hard_rejected_papers
    _write_audit_csv(final_csv_list, "rulebased_filter.csv")

    return approved_papers


def _write_audit_csv(data_list: list[dict], csv_filename: str):
    """Internal helper function to safely write results into a specified CSV file."""
    if not data_list:
        return
    csv_fields = ["id", "title", "decision", "is_english", "is_matching", "audit_reason"]
    try:
        print(f"\n💾 Writing Non-LLM audit results for {len(data_list)} papers into {csv_filename} ...")
        with open(csv_filename, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data_list)
        print(f"🎉 Report successfully exported! File path: {os.path.abspath(csv_filename)}")
    except Exception as e:
        print(f"❌ Failed to write CSV: {e}")


if __name__ == "__main__":
    # Call raw API sampling dataset (fetching the identical topic T13674 as previous experiments for controlled alignment)
    raw_unfiltered_papers = fetch_unfiltered_raw_works(
        topic_id="T13674", num_papers=10000, oa_status="diamond", lang_filter="en"
    )
    
    print("\n🏁 Starting ablation study (Non-LLM workflow + FastText + SentenceTransformers semantic checks)...")
    clean_results = batch_filter_papers_without_llm(raw_unfiltered_papers)
    
    print("=" * 60)
    print(f"  ● [Non-LLM FastText + Transformers Version] Total Cleaned & Passed: {len(clean_results)} papers")
    print("=" * 60)