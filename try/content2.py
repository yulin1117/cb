import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util

def analyze_semantic_threshold(csv_path="data_cleaning_audit_report.csv", 
                               output_csv="failed_matches_similarity_check.csv", 
                               output_md="content_check.md", 
                               threshold=0.5):
    """
    Evaluates the compliance of LLM 'is_matching=False' rejections against 
    a fast SentenceTransformer vector similarity gate.
    """
    if not os.path.exists(csv_path):
        print(f"❌ Error: Source audit file not found at: {csv_path}")
        return

    print(f"📖 Reading source audit database: {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. OpenAlex abstracts are inverted indexes. Your audit file might already contain 
    # a 'reconstructed_abstract' column or have it stored within the 'audit_reason'.
    # If your CSV uses a specific column name for abstract text, modify this line:
    abstract_col = "abstract" if "abstract" in df.columns else "audit_reason"

    # 2. Filter data: strict isolation of papers rejected by the LLM matching pipeline
    # We ensure values are cast cleanly to string and stripped of system noise
    df_filtered = df[df["is_matching"].astype(str).str.strip().str.upper() == "FALSE"].copy()
    
    if df_filtered.empty:
        print("🎉 Absolute Consensus! No records found with 'is_matching=False'.")
        return

    total_records = len(df_filtered)
    print(f"📊 Isolated {total_records:,} papers rejected by LLM for vector similarity auditing...")

    # 3. Initialize the Vector Embedding Engine
    print("🤖 Loading SentenceTransformer ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Extract raw clean text components for batch operations
    titles = df_filtered["title"].fillna("").astype(str).str.strip().tolist()
    abstracts = df_filtered[abstract_col].fillna("").astype(str).str.strip().tolist()

    # 🚀 Matrix Vectorized Encoding (Highly optimized parallel matrix tracking)
    print(f"⚡ Batch vectorizing text features using multi-core PyTorch allocations...")
    t_embeddings = model.encode(titles, batch_size=64, show_progress_bar=True)
    a_embeddings = model.encode(abstracts, batch_size=64, show_progress_bar=True)

    # 4. Process comparisons pairwise via NumPy fast evaluation layers
    print("🧮 Calculating Cosine Similarity profiles and decision matching vectors...")
    sim_scores = []
    vector_decisions = []
    is_matching_same_list = []

    for i in range(total_records):
        # Calculate fast cosine similarity metrics
        score = util.cos_sim(t_embeddings[i], a_embeddings[i]).item()
        sim_scores.append(score)
        
        # Rule-based decision profile configuration
        rule_decision = False if score < threshold else True
        vector_decisions.append(rule_decision)
        
        # Ground Truth comparison: LLM matching is False. 
        # If the vector similarity also marks it as False (score < threshold), they match!
        is_same = (rule_decision == False)
        is_matching_same_list = is_matching_same_list + [is_same]

    # Append computational profiles into our evaluated frame
    df_filtered["similarity_score"] = sim_scores
    df_filtered["vector_is_matching"] = vector_decisions
    df_filtered["is_matching_same"] = is_matching_same_list

    # Export structured sub-matrix evaluation CSV
    out_fields = ["id", "title", "decision", "is_matching", "similarity_score", "vector_is_matching", "is_matching_same"]
    df_filtered[out_fields].to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"💾 Discrepancy dataset safely written to: {os.path.abspath(output_csv)}")

    # 5. Calculate Consensus Rate and Diagnostics Statistics
    true_consensus_count = sum(is_matching_same_list)
    consensus_rate = (true_consensus_count / total_records) * 100
    
    # Track cases where the threshold didn't catch the LLM rejection (over-lenient threshold)
    lenient_leakage = total_records - true_consensus_count 

    # Generate Analytical MD Report
    md_report = f"""# Structural Semantic Alignment Diagnostics Report
This validation assessment traces the behavioral alignment between **LLM Semantic Filtering** and **Bi-Encoder Vector Similarities** applied explicitly to documents flagged as mismatched (`is_matching=False`).

---

## 1. Threshold Benchmark Metrics
- **Target Calibration Dataset Size**: {total_records:,} papers
- **Evaluated Vector Pipeline Threshold**: `{threshold}`
- **Identical Pipeline Decisions (Consensus)**: {true_consensus_count:,} papers
- **Decision Alignment Rate (Consensus Rate)**: **{consensus_rate:.2f}%**

---

## 2. Decision Distribution Matrix
| Evaluation Profile | Measurement Count | System Operational Meaning |
| :--- | :---: | :--- |
| **True Negative Alignment** | {true_consensus_count:,} | Both LLM and Vector Pipeline agree the paper is a mismatch (Score < {threshold}). |
| **System Leakage Vulnerability** | {lenient_leakage:,} | LLM rejected the paper, but the Vector Pipeline passed it (Score >= {threshold}). |

---

## 3. Heuristic Threshold Calibration Insight
### Current Observation
- Out of {total_records:,} documents that the LLM explicitly flagged as garbage or out-of-domain noise, the vector evaluation pipeline confirmed **{consensus_rate:.2f}%** of them using a threshold value of `{threshold}`.
"""
    if consensus_rate >= 90.0:
        md_report += f"\n> 🟢 **Recommendation:** The alignment rate is highly optimal ({consensus_rate:.2f}%). The default threshold of `{threshold}` is mathematically robust enough to replicate LLM behavior without burning GPU cluster tokens.\n"
    else:
        # Dynamic advice based on mathematical profiling
        mean_score_of_leaks = df_filtered[df_filtered["is_matching_same"] == False]["similarity_score"].mean()
        md_report += f"\n> ⚠️ **Recommendation:** The current threshold of `{threshold}` might be too lenient, allowing {lenient_leakage:,} noisy records to slip through. The average score of these leaked items is **{mean_score_of_leaks:.4f}**. Consider bumping the threshold parameter up to **{(mean_score_of_leaks + 0.05):.2f}** to tighten down structural security boundaries.\n"

    md_report += "\n---\n*Report dynamically generated via check_semantic_threshold.py.*\n"

    with open(output_md, mode="w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"💾 Diagnostic analytics report successfully saved to: {os.path.abspath(output_md)}")

if __name__ == "__main__":
    analyze_semantic_threshold()