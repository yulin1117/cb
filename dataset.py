import os
import pandas as pd
import pycountry
import matplotlib.pyplot as plt
import json
import csv


class Dataset:
    def __init__(self):
        self.raw_dir = "data/raw"
        self.data_dir = "data"
        self.df = None
        self.context_docs = None

    def create(self, topic: str):
        """
        Load raw CSV from data/raw, clean it, and save the final cleaned CSV to data/
        The filename remains the same. Only the top 100 papers by Relevance-Score are saved.
        """
        safe_topic = topic.replace(" ", "_")
        raw_file = os.path.join(self.raw_dir, f"{safe_topic}.csv")
        final_file = os.path.join(self.data_dir, f"{safe_topic}.csv")

        if not os.path.exists(raw_file):
            raise FileNotFoundError(f"Raw CSV for topic '{topic}' not found at {raw_file}")

        # Load raw CSV
        df = pd.read_csv(
            raw_file,
            engine='python',
            encoding='utf-8',
            quoting=csv.QUOTE_MINIMAL,
            on_bad_lines='skip'
        )

        # Column mapping
        columns = {
            "id": "Paper-ID",
            "relevance_score": "Relevance-Score",
            "title": "Title",
            "abstract": "Abstract",
            "publication_year": "Publication-Year",
            "type": "Paper-Type",
            "cited_by_count": "Citation-Count",
            "primary_location.raw_source_name": "Venue",
            "institutions.display_name": "Institution",
            "authorships.author.display_name": "Author",
            "authorships.countries": "Country"
        }

        # Select and rename columns
        selected = [col for col in columns.keys() if col in df.columns]
        df = df[selected].copy()
        df = df.rename(columns=columns)

        # Remove OpenAlex prefix
        if "Paper-ID" in df.columns:
            df["Paper-ID"] = df["Paper-ID"].astype(str).str.replace("https://openalex.org/", "")

        # Clean multi-label fields
        for col in ["Institution", "Author", "Country"]:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace("|", ", ")
                    .str.replace(r"\s*,\s*", ", ", regex=True)
                    .str.strip(", ")
                )
                df[col] = df[col].replace({"": "nan", ",": "nan", ", ,": "nan", "nan,": "nan", ", nan": "nan"})

        # Convert countries
        if "Country" in df.columns:
            def expand_countries(value):
                if pd.isna(value) or value.lower() == "nan":
                    return "nan"
                countries = [c.strip() for c in value.split(",") if c.strip()]
                expanded = []
                for c in countries:
                    try:
                        country_obj = pycountry.countries.get(alpha_2=c)
                        if country_obj:
                            country_name = country_obj.name
                            if "," in country_name:
                                parts = [p.strip() for p in country_name.split(",")]
                                country_name = " ".join(parts[1:] + [parts[0]])
                            expanded.append(country_name)
                        else:
                            expanded.append(c)
                    except Exception:
                        expanded.append(c)
                seen = set()
                unique_countries = [x for x in expanded if not (x in seen or seen.add(x))]
                return ", ".join(unique_countries) if unique_countries else "nan"

            df["Country"] = df["Country"].apply(expand_countries)

        # Normalize Paper-Type
        if "Paper-Type" in df.columns:
            df["Paper-Type"] = df["Paper-Type"].astype(str).apply(
                lambda x: "-".join(part.capitalize() for part in x.split("-")) if x.lower() != "nan" else "nan"
            )

        # Convert numeric fields robustly
        if "Publication-Year" in df.columns:
            def clean_year(x):
                if pd.isna(x):
                    return "nan"
                try:
                    return str(int(float(x)))
                except (ValueError, TypeError):
                    return "nan"

            df["Publication-Year"] = df["Publication-Year"].apply(clean_year)

        if "Citation-Count" in df.columns:
            def clean_count(x):
                if pd.isna(x):
                    return "nan"
                try:
                    return str(int(float(x)))
                except (ValueError, TypeError):
                    return "nan"

            df["Citation-Count"] = df["Citation-Count"].apply(clean_count)

        # Convert Relevance-Score to numeric safely
        if "Relevance-Score" in df.columns:
            df["Relevance-Score"] = pd.to_numeric(df["Relevance-Score"], errors="coerce")
            df = df[df["Relevance-Score"].notna()]

        # Filter Abstract
        if "Abstract" in df.columns:
            df = df[df["Abstract"].notna()]

        # Remove rows with any 'nan'
        df = df[~df.apply(lambda row: row.astype(str).str.contains(r"\bnan\b", case=False)).any(axis=1)]

        # Select top 100 by Relevance-Score
        if "Relevance-Score" in df.columns:
            df = df.nlargest(100, "Relevance-Score")

        # Ensure output directory exists
        os.makedirs(self.data_dir, exist_ok=True)

        # Save cleaned CSV
        df.to_csv(final_file, index=False)
        print(f"Cleaned top 100 papers saved to {final_file}")

    def load(self, topic: str, top_n: int = 20, biases: list[str] | None = None):
        """
        Load the cleaned CSV and prepare context documents with selected features.
        Automatically calls create() if the cleaned dataset does not exist.
        """
        if biases is None:
            biases = []

        # Ensure "title" and "abstract" are first
        if "title" not in biases:
            biases.insert(0, "Title")
        if "abstract" not in biases:
            biases.insert(1, "Abstract")

        safe_topic = topic.replace(" ", "_")
        file_path = os.path.join(self.data_dir, f"{safe_topic}.csv")

        # If cleaned CSV does not exist, create it
        if not os.path.exists(file_path):
            print(f"Cleaned dataset for topic '{topic}' not found. Creating now...")
            self.create(topic)

        # Load the cleaned CSV
        df = pd.read_csv(file_path)

        # Store full cleaned df
        self.df = df.copy()

        # Prepare context_docs using only requested features
        feature_to_col = {bias: bias.capitalize() if bias in ["title", "abstract"] else bias for bias in biases}
        id_col = "Paper-ID"

        # Select top_n by Relevance-Score
        if "Relevance-Score" in df.columns:
            df_top = df.nlargest(top_n, "Relevance-Score")
        else:
            df_top = df.copy()

        # Prepare context_docs
        context_docs = {}
        for _, row in df_top.iterrows():
            if id_col not in row or pd.isna(row[id_col]):
                continue
            doc_id = str(row[id_col])
            bias_map = {}
            for bias in biases:
                mapped_col = feature_to_col.get(bias, bias)
                if mapped_col in row and not pd.isna(row[mapped_col]):
                    bias_map[bias] = str(row[mapped_col])
            context_docs[doc_id] = bias_map

        self.context_docs = context_docs
        return context_docs


def get_bias_values(topics: list[str], bias: str, top_n: int = 20, plot: bool = False):
    bias_values = dict()

    for topic in topics:
        data = Dataset().load(topic=topic, biases=[bias], top_n=top_n)

        # data structure: {Paper-ID: {"Title": t, "Abstract": a, "bias": b}}
        for paper_id, paper_data in data.items():
            b = paper_data.get(bias)
            if b is not None:
                bias_values[b] = bias_values.get(b, 0) + 1

    # Try converting bias values to numeric if possible
    converted = {}
    numeric = True
    for k, v in bias_values.items():
        try:
            converted[float(k)] = v
        except (ValueError, TypeError):
            numeric = False
            break

    # Sort and reassign depending on type
    if numeric:
        bias_values = dict(sorted(converted.items(), key=lambda x: x[0]))
    else:
        bias_values = dict(sorted(bias_values.items(), key=lambda x: str(x[0])))

    # Plot if requested
    if plot and bias_values:
        plt.figure(figsize=(10, 4))
        if numeric:
            # Line plot for numerical values
            plt.plot(list(bias_values.keys()), list(bias_values.values()), marker="o", linestyle="-")
            plt.title(f"Numerical Bias Distribution for '{bias}'")
            plt.xlabel("Bias Value")
            plt.ylabel("Count")
        else:
            # Bar plot for categorical values
            plt.bar(bias_values.keys(), bias_values.values())
            plt.title(f"Categorical Bias Distribution for '{bias}'")
            plt.xlabel("Bias Category")
            plt.ylabel("Count")
            plt.xticks(rotation=90, ha="right")

        plt.tight_layout()
        plt.show()

    return bias_values


def load_results(prompt, topic, model: str, bias: str | None = None):
    """
    Load results for a given prompt, topic, and model, optionally filtered by bias.
    :param prompt: Prompt class instance
    :param topic: Topic string
    :param model: Model name used in run_task
    :param bias: Optional bias string
    :return: Loaded results as a list
    """
    safe_topic = topic.replace(" ", "-")
    safe_model = str(model).replace("/", "_").replace(":", "_")
    model_dir = os.path.join("out", safe_model)

    if bias is None:
        filename = f"{prompt.name}_{safe_topic}.json"
    else:
        filename = f"{prompt.name}_{safe_topic}_{bias}.json"

    file_path = os.path.join(model_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Results file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    return results


