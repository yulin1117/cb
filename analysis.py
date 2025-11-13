from dataset import load_results, Dataset

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, f_oneway, kruskal


def calculate_biases(prompt, topics, model, bias: str):
    all_rows = []

    for topic in topics:
        # Load results for this topic
        unbiased_results = load_results(prompt=prompt, topic=topic, model=model, bias=None)
        biased_results = load_results(prompt=prompt, topic=topic, model=model, bias=bias)

        # Load paper metadata for this topic
        data = Dataset().load(topic=topic, biases=[bias])
        all_paper_ids = sorted(set(data.keys()))

        def expand_results(results, condition):
            rows = []
            for r in results:
                iteration = r["iteration"]
                cited_set = set(r["citations"])
                for pid in all_paper_ids:
                    meta = data.get(pid, {})
                    meta_value = meta.get(bias, None)

                    # Handle multi-valued metadata
                    if isinstance(meta_value, str):
                        meta_values = [v.strip() for v in meta_value.split(",") if v.strip()]
                    elif isinstance(meta_value, list):
                        meta_values = meta_value
                    else:
                        meta_values = [meta_value]

                    for val in meta_values:
                        rows.append({
                            "topic": topic,
                            "iteration": iteration,
                            "condition": condition,
                            "paper_id": pid,
                            "selected": int(pid in cited_set),
                            "metadata": val,
                            "title": meta.get("Title", None),
                        })
            return rows

        # Expand and append rows for this topic
        all_rows.extend(expand_results(unbiased_results, "unbiased"))
        all_rows.extend(expand_results(biased_results, "biased"))

    # Combine all topics into a single DataFrame
    df = pd.DataFrame(all_rows)

    # ------------------------------------------------------------
    # Compute selection deltas per paper
    # ------------------------------------------------------------
    def compute_bias_effects(df):
        paper_summary = (
            df.groupby(["condition", "paper_id", "metadata"])
            .agg(selection_frequency=("selected", "mean"),
                 title=("title", "first"))
            .reset_index()
        )

        # Pivot by condition to compute deltas
        paper_wide = (paper_summary
                      .pivot_table(index=["paper_id", "metadata"], columns="condition", values="selection_frequency")
                      .reset_index())

        paper_wide["delta_selection"] = paper_wide["biased"] - paper_wide["unbiased"]
        return paper_wide

    # ------------------------------------------------------------
    # Correlation / Statistical testing
    # ------------------------------------------------------------
    def analyze_bias(paper_wide, bias: str):
        print(f"\n=== Bias Analysis for: {bias} ===")

        def significance_label(p):
            return " (significant)" if p < 0.05 else " (not significant)"

        # Convert metadata to numeric if possible
        paper_wide = paper_wide.copy()
        paper_wide["metadata_numeric"] = pd.to_numeric(paper_wide["metadata"], errors='coerce')
        numeric_df = paper_wide.dropna(subset=["metadata_numeric"])

        # Numeric bias analysis
        if pd.api.types.is_numeric_dtype(numeric_df["metadata_numeric"]) and len(numeric_df) >= 2:
            x = numeric_df["metadata_numeric"]
            y = numeric_df["delta_selection"]

            pearson_corr, pearson_p = pearsonr(x, y)
            spearman_corr, spearman_p = spearmanr(x, y)

            print(f"Pearson r = {pearson_corr:.3f} (p={pearson_p:.3g}){significance_label(pearson_p)}")
            print(f"Spearman ρ = {spearman_corr:.3f} (p={spearman_p:.3g}){significance_label(spearman_p)}")

            plt.figure(figsize=(6, 4))
            sns.regplot(x=x, y=y, scatter_kws={'alpha': 0.6})
            plt.xlabel(f"{bias} (numeric)")
            plt.ylabel("Δ Selection Frequency (biased - unbiased)")
            plt.title(f"Effect of {bias} on LLM citation behavior")

            # Baseline at y=0 (same as categorical)
            plt.axhline(0, color='black', linestyle='--', linewidth=1)

            plt.tight_layout()
            plt.show()

        else:
            plt.figure(figsize=(10, 5))

            # Violin plot showing distribution
            sns.violinplot(
                x="metadata", y="delta_selection", data=paper_wide,
                inner=None, alpha=0.7
            )

            # Overlay category means (using modern parameters)
            sns.pointplot(
                x="metadata", y="delta_selection", data=paper_wide,
                estimator="mean", color="black",
                markers="D", linestyle="none",  # replaces join=False
                err_kws={"linewidth": 0},  # replaces errwidth=0
                markersize=6  # replaces scale
            )

            # Baseline at y=0
            plt.axhline(0, color='black', linestyle='--', linewidth=1)

            plt.xlabel(bias)
            plt.ylabel("Δ Selection Frequency (biased - unbiased)")
            plt.title(f"Effect of categorical bias: {bias}")
            plt.xticks(rotation=90, ha="right")
            plt.tight_layout()
            plt.show()

            # Statistical significance test (ANOVA + Kruskal–Wallis)
            groups = [grp["delta_selection"].values for _, grp in paper_wide.groupby("metadata") if len(grp) > 1]
            if len(groups) > 1:
                try:
                    f_stat, p_val = f_oneway(*groups)
                    h_stat, h_p_val = kruskal(*groups)

                    print(f"\nANOVA: F={f_stat:.3f}, p={p_val:.3g}{significance_label(p_val)}")
                    print(f"Kruskal–Wallis: H={h_stat:.3f}, p={h_p_val:.3g}{significance_label(h_p_val)}")
                except Exception as e:
                    print(f"\nStatistical test failed: {e}")

    paper_wide = compute_bias_effects(df)
    analyze_bias(paper_wide, bias=bias)
