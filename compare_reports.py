import os
import csv
import pandas as pd
import numpy as np

def run_ablation_study_analysis(llm_csv="data_cleaning_audit_report.csv", 
                                no_llm_csv="rulebased_filter.csv", 
                                output_md="rule_vs_llm_content_check.md"):
    """
    Reads the LLM-based audit report and the non-LLM (Rule-based) audit report,
    performs ablation study comparisons, prints professional English diagnostics,
    generates a comprehensive Markdown report, and exports the inconsistent papers
    into two dedicated CSV files for manual inspection.
    """
    if not os.path.exists(llm_csv) or not os.path.exists(no_llm_csv):
        print(f"❌ Error: Missing files for comparison. Please ensure '{llm_csv}' and '{no_llm_csv}' exist.")
        return

    # 1. Load datasets
    df_llm = pd.read_csv(llm_csv)
    df_no_llm = pd.read_csv(no_llm_csv)

    total_llm = len(df_llm)
    total_no_llm = len(df_no_llm)

    # 2. Extract statistics
    llm_counts = df_llm["decision"].value_counts()
    no_llm_counts = df_no_llm["decision"].value_counts()

    llm_approved = llm_counts.get("APPROVED", 0)
    llm_rejected = llm_counts.get("REJECTED", 0)
    llm_app_rate = (llm_approved / total_llm) * 100 if total_llm > 0 else 0.0

    no_llm_approved = no_llm_counts.get("APPROVED", 0)
    no_llm_rejected = no_llm_counts.get("REJECTED", 0)
    no_llm_app_rate = (no_llm_approved / total_no_llm) * 100 if total_no_llm > 0 else 0.0

    strictness_conclusion = ""
    if no_llm_app_rate < llm_app_rate:
        strictness_conclusion = "The [Non-LLM Rule-Based Pipeline] is MORE strict (lower acceptance rate) than the LLM counterpart."
    elif no_llm_app_rate > llm_app_rate:
        strictness_conclusion = "The [LLM-Based Pipeline] is MORE strict (lower acceptance rate), filtering out more subtle semantic noise."
    else:
        strictness_conclusion = "Both pipelines exhibit identical strictness levels."

    # 3. Pairwise Alignment using Inner Join
    # Keep key columns from LLM and bring in the decision and reason from No-LLM
    df_llm_sub = df_llm[["id", "title", "decision", "audit_reason"]].rename(
        columns={"decision": "decision_llm", "audit_reason": "audit_reason_llm"}
    )
    df_no_llm_sub = df_no_llm[["id", "decision", "audit_reason"]].rename(
        columns={"decision": "decision_no_llm", "audit_reason": "audit_reason_no_llm"}
    )
    
    merged = pd.merge(df_llm_sub, df_no_llm_sub, on="id", how="inner")

    if merged.empty:
        print("⚠️ Warning: No aligned papers found after merging. Check if 'id' formats match.")
        return

    total_aligned = len(merged)

    # Calculate Confusion Matrix Quadrants
    tp_df = merged[(merged["decision_llm"] == "APPROVED") & (merged["decision_no_llm"] == "APPROVED")]
    tn_df = merged[(merged["decision_llm"] == "REJECTED") & (merged["decision_no_llm"] == "REJECTED")]
    
    # 🚨 Case 1: Rule Approved but LLM Rejected (Leakage / False Positive)
    fp_df = merged[(merged["decision_llm"] == "REJECTED") & (merged["decision_no_llm"] == "APPROVED")]
    
    # 🚨 Case 2: Rule Rejected but LLM Approved (Overkill / False Negative)
    fn_df = merged[(merged["decision_llm"] == "APPROVED") & (merged["decision_no_llm"] == "REJECTED")]

    tp, tn, fp, fn = len(tp_df), len(tn_df), len(fp_df), len(fn_df)
    consensus_rate = ((tp + tn) / total_aligned) * 100 if total_aligned > 0 else 0.0

    # ──────────────────────────────────────────────────────────────────────
    # 💾 NEW: Export Inconsistent Subsets to Dedicated CSV Files
    # ──────────────────────────────────────────────────────────────────────
    csv_out_fields = ["id", "title", "decision_llm", "audit_reason_llm", "decision_no_llm", "audit_reason_no_llm"]
    
    fp_csv_path = "rule_approved_llm_reject.csv"
    fn_csv_path = "rule_reject_llm_approved.csv"

    # Export Leakages (Rule Approved, LLM Rejected)
    fp_df[csv_out_fields].to_csv(fp_csv_path, index=False, encoding="utf-8-sig")
    # Export Overkills (Rule Rejected, LLM Approved)
    fn_df[csv_out_fields].to_csv(fn_csv_path, index=False, encoding="utf-8-sig")

    # 4. Print English Report to Console
    console_border = "=" * 80
    print(console_border)
    print("                 📋 ABLATION STUDY REAL-TIME REPORT 📋")
    print(console_border)
    print(f"Loaded successfully:")
    print(f"  ● LLM-Based Dataset Records     : {total_llm:,} papers")
    print(f"  ● Non-LLM Rule-Based Dataset   : {total_no_llm:,} papers")
    print(f"  ● Successfully Aligned Pairwise : {total_aligned:,} papers")
    print(console_border)
    print("💡 SECTION 1: PASS RATE & STRICTNESS COMPARISON")
    print("-" * 80)
    print(f"🤖 [LLM Pipeline]      APPROVED: {llm_approved:<5} | REJECTED: {llm_rejected:<5} | Pass Rate: {llm_app_rate:.2f}%")
    print(f"⚡ [Rule-Based (No-LLM)] APPROVED: {no_llm_approved:<5} | REJECTED: {no_llm_rejected:<5} | Pass Rate: {no_llm_app_rate:.2f}%")
    print(f"\nConclusion: {strictness_conclusion}")
    print(console_border)
    print("💡 SECTION 2: PAIRWISE DECISION CONFUSION MATRIX")
    print("-" * 80)
    print("Using [LLM Pipeline] as Ground Truth:")
    print("\n                         [Rule-Based] Predicted APPROVED  [Rule-Based] Predicted REJECTED")
    print(f"[LLM] Ground APPROVED  : {tp:<30} (True Consensus) {fn:<30} (Overkill)")
    print(f"[LLM] Ground REJECTED  : {fp:<30} (Leakage)        {tn:<30} (False Consensus)")
    print("-" * 80)
    print(f"📊 Overall Decision Consensus Rate: {consensus_rate:.2f}%")
    print(console_border)
    print("💡 SECTION 3: INCONSISTENCY CSV EXPORT SUMMARY")
    print("-" * 80)
    print(f"💾 1. Leakages (Rule App / LLM Rej) exported to: {os.path.abspath(fp_csv_path)} ({fp} rows)")
    print(f"💾 2. Overkills (Rule Rej / LLM App) exported to: {os.path.abspath(fn_csv_path)} ({fn} rows)")
    print(console_border)

    # 5. Build Markdown Content
    md_content = f"""# Ablation Study Report: LLM vs. Non-LLM Pipeline
This report presents a pairwise comparison between the **LLM-Based Pipeline** and the **Non-LLM Rule-Based Pipeline** (integrated with FastText & Stopwords Radar heuristics) on OpenAlex scholarly documents.

---

## 1. Pipeline Dataset Statistics
- **Total Records (LLM Pipeline)**: {total_llm:,}
- **Total Records (Rule-Based Pipeline)**: {total_no_llm:,}
- **Successfully Aligned Pairwise Records**: {total_aligned:,}

---

## 2. Pass Rate & Strictness Comparison
| Cleaning Pipeline | Approved Papers | Rejected Papers | Pass Rate (%) |
| :--- | :---: | :---: | :---: |
| **🤖 LLM Pipeline** | {llm_approved} | {llm_rejected} | {llm_app_rate:.2f}% |
| **⚡ Non-LLM Rule-Based** | {no_llm_approved} | {no_llm_rejected} | {no_llm_app_rate:.2f}% |

### Strictness Analysis
> **Conclusion:** {strictness_conclusion}

---

## 3. Aligned Confusion Matrix
Using the **LLM Pipeline** as the reference standard (Ground Truth), the classification matrix of the **Rule-Based Pipeline** is detailed below:

| | Rule-Based APPROVED (Predicted Positive) | Rule-Based REJECTED (Predicted Negative) |
| :--- | :---: | :---: |
| **LLM APPROVED (Actual Positive)** | **{tp}** <br>*(True Consensus / APPROVED in Both)* | **{fn}** <br>*(Overkill / False Negative)* <br>👉 See `{fn_csv_path}` |
| **LLM REJECTED (Actual Negative)** | **{fp}** <br>*(Leakage / False Positive)* <br>👉 See `{fp_csv_path}` | **{tn}** <br>*(False Consensus / REJECTED in Both)* |

### Summary Statistic
- **Pairwise Consensus Rate**: **{consensus_rate:.2f}%**

---

## 4. Extreme Diagnostic Case Inspection (Samples)

### 🚨 Case A: Pipeline Leakage (FP Count: {fp})
*These are documents rejected by LLM's deep semantic understanding but passed by the physical heuristic rules. Full list in `{fp_csv_path}`.*
"""
    if fp > 0:
        for idx, row in fp_df.head(5).iterrows():
            md_content += f"- **ID**: `{row['id']}` | **Title**: {row['title']}\n"
    else:
        md_content += "_Perfect consensus! The rule-based pipeline successfully captured all rejections without any leakages._\n"

    md_content += f"""
### 🎯 Case B: Pipeline Overkill (FN Count: {fn})
*These are documents marked as APPROVED by LLM, but prematurely filtered out by the Rule-Based Pipeline. Full list in `{fn_csv_path}`.*
"""
    if fn > 0:
        for idx, row in fn_df.head(5).iterrows():
            md_content += f"- **ID**: `{row['id']}` | **Title**: {row['title']}\n"
    else:
        md_content += "_Excellent! No overkill detected. Clean papers were completely unharmed by the rule filters._\n"

    md_content += "\n---\n*Report dynamically generated via compare_reports.py.*\n"

    with open(output_md, mode="w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"💾 Markdown report successfully saved to: {os.path.abspath(output_md)}")

if __name__ == "__main__":
    run_ablation_study_analysis()