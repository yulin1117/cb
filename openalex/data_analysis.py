from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from openalex.openalex import (
    _get_fasttext_model,
    _get_specter_model,
    abstract_lang_detect,
    content_detect,
    duplicate_detect,
    reconstruct_abstract,
    title_lang_detect,
)


def _strip_openalex_id(url_or_id: str) -> str:
    return str(url_or_id).rstrip("/").split("/")[-1]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for v in values:
        if not v:
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _join_pipe(values: list[str]) -> str:
    values = [str(v).strip() for v in values if str(v).strip()]
    values = _dedupe_keep_order(values)
    return "|".join(values)


def _extract_venue(work: dict[str, Any]) -> str:
    primary_location = work.get("primary_location") or {}
    if not isinstance(primary_location, dict):
        return ""

    for value in [
        primary_location.get("raw_source_name"),
        (
            (primary_location.get("source") or {}).get("display_name")
            if isinstance(primary_location.get("source"), dict)
            else None
        ),
    ]:
        if value:
            return str(value)

    return ""


def _extract_authors(work: dict[str, Any]) -> str:
    authorships = work.get("authorships") or []
    authors = []

    for authorship in authorships:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author") or {}
        if not isinstance(author, dict):
            continue
        name = author.get("display_name")
        if name:
            authors.append(str(name))

    return _join_pipe(authors)


def _extract_institutions(work: dict[str, Any]) -> str:
    authorships = work.get("authorships") or []
    institutions = []

    for authorship in authorships:
        if not isinstance(authorship, dict):
            continue
        for inst in authorship.get("institutions") or []:
            if not isinstance(inst, dict):
                continue
            name = inst.get("display_name")
            if name:
                institutions.append(str(name))

    return _join_pipe(institutions)


def _extract_countries(work: dict[str, Any]) -> str:
    authorships = work.get("authorships") or []
    countries = []

    for authorship in authorships:
        if not isinstance(authorship, dict):
            continue

        auth_countries = authorship.get("countries")
        if isinstance(auth_countries, list):
            countries.extend([str(c) for c in auth_countries if c])

        for inst in authorship.get("institutions") or []:
            if not isinstance(inst, dict):
                continue
            country_code = inst.get("country_code")
            if country_code:
                countries.append(str(country_code))

    return _join_pipe(countries)


def _fetch_work_from_openalex(
    paper_id: str,
    mailto: str = "tobias.schreieder@tu-dresden.de",
    timeout: int = 60,
    max_retries: int = 6,
) -> dict[str, Any] | None:
    work_id = _strip_openalex_id(paper_id)
    url = f"https://api.openalex.org/works/{work_id}"

    headers = {
        "User-Agent": f"CitationBiasBenchmark (mailto:{mailto})",
    }

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)

            if resp.status_code == 200:
                work = resp.json()
                if not isinstance(work, dict):
                    print(f"Warning: unexpected response for {work_id}")
                    return None

                inv = work.get("abstract_inverted_index") or work.get(
                    "abstract_inversted_index"
                )
                if inv:
                    try:
                        work["abstract"] = reconstruct_abstract(inv)
                    except Exception as e:
                        print(
                            f"Warning: failed to reconstruct abstract for {work_id}: {e}"
                        )
                        work["abstract"] = ""
                else:
                    work["abstract"] = ""

                return work

            if resp.status_code in (429, 500, 502, 503, 504):
                sleep_s = min(60, 2**attempt)
                time.sleep(sleep_s)
                continue

            print(f"Warning: failed to fetch {work_id} ({resp.status_code})")
            return None

        except requests.RequestException as e:
            if attempt == max_retries - 1:
                print(f"Warning: request failed for {work_id}: {e}")
                return None
            sleep_s = min(60, 2**attempt)
            time.sleep(sleep_s)

    return None


def _work_to_row(
    work: dict[str, Any],
    field: str,
    topic_name: str,
) -> dict[str, Any]:
    work_id = _strip_openalex_id(work.get("id", ""))

    return {
        "field": field,
        "topic": topic_name,
        "id": work_id,
        "title": work.get("title", "") or "",
        "abstract": work.get("abstract", "") or "",
        "year": work.get("publication_year", "") or "",
        "type": work.get("type", "") or "",
        "citation_count": work.get("cited_by_count", "") or "",
        "venue": _extract_venue(work),
        "institution": _extract_institutions(work),
        "author": _extract_authors(work),
        "country": _extract_countries(work),
    }


def dataset_to_csv(
    topics_base_dir: str | Path = "../out/topics",
    output_csv: str | Path = "../out/openalex/papers.csv",
    mailto: str = "tobias.schreieder@tu-dresden.de",
    sleep_seconds: float = 0.1,
    deduplicate_papers: bool = True,
) -> Path:
    """
    Build one CSV at:
        out/openalex/papers.csv

    Workflow:
    1. Scan subdirectories in `topics_base_dir` as `topic_id`.
    2. Load existing CSV if present to cache fetched paper metadata.
    3. For each topic folder:
       - Load `topicgpt_labels.json` for umbrella field and topic metadata.
       - Load `final_verified_papers.json` for the list of verified papers.
       - Fetch each paper from the OpenAlex API (or load from cache).
       - Flatten metadata fields into CSV row format.

    Final CSV columns:
        field, topic, id, title, abstract, year, type,
        citation_count, venue, institution, author, country
    """
    topics_base_dir = Path(topics_base_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load existing CSV if it exists to avoid re-fetching papers
    existing_papers: dict[str, dict[str, Any]] = {}
    if output_csv.exists():
        print(f"Loading existing CSV from {output_csv} to avoid re-fetching papers...")
        try:
            existing_df = pd.read_csv(output_csv, dtype=str)
            for _, r in existing_df.iterrows():
                pid = _strip_openalex_id(r.get("id", ""))
                if pid:
                    existing_papers[pid] = r.to_dict()
            print(f"Loaded {len(existing_papers)} existing papers from cache.")
        except Exception as e:
            print(f"Warning: Failed to load existing CSV ({e}). Will fetch all papers from API.")

    if not topics_base_dir.exists():
        print(f"Error: Base directory {topics_base_dir} does not exist.")
        return output_csv

    topic_dirs = sorted([d for d in topics_base_dir.iterdir() if d.is_dir()])
    rows: list[dict[str, Any]] = []
    seen_papers: set[str] = set()

    for topic_dir in topic_dirs:
        topic_id = topic_dir.name
        reps_path = topic_dir / "final_verified_papers.json"
        labels_path = topic_dir / "topicgpt_labels.json"

        if not reps_path.exists():
            print(f"Skipping {topic_id}: missing {reps_path}")
            continue

        if not labels_path.exists():
            print(f"Skipping {topic_id}: missing {labels_path}")
            continue

        try:
            reps = _load_json(reps_path)
            labels = _load_json(labels_path)
        except Exception as e:
            print(f"Skipping {topic_id}: failed to load JSON files ({e})")
            continue

        # Extract 'field' (umbrella topic) and 'topic_name'
        field = labels.get("umbrella_display_name", "") or topic_id

        cluster_result = labels.get("cluster_result", {})
        if isinstance(cluster_result, dict):
            topic_name = cluster_result.get("topic_name", "")
        else:
            topic_name = ""

        if not topic_name:
            topic_name = reps.get("topic_name", "")

        if not topic_name:
            print(f"Skipping {topic_id}: missing topic_name in JSON files")
            continue

        papers = reps.get("papers", [])
        if not isinstance(papers, list):
            print(f"Skipping {topic_id}: 'papers' field is not a list in {reps_path}")
            continue

        print(f"Processing topic {topic_id} ({topic_name}) with {len(papers)} papers...")

        for paper in papers:
            if not isinstance(paper, dict):
                continue

            paper_id = _strip_openalex_id(paper.get("id", ""))
            if not paper_id:
                continue

            if deduplicate_papers and paper_id in seen_papers:
                continue
            seen_papers.add(paper_id)

            # Check if paper is already in existing CSV cache
            if paper_id in existing_papers:
                cached_row = existing_papers[paper_id].copy()
                cached_row["field"] = field
                cached_row["topic"] = topic_name
                rows.append(cached_row)
                continue

            work = _fetch_work_from_openalex(
                paper_id=paper_id,
                mailto=mailto,
            )
            if not work:
                continue

            row = _work_to_row(
                work=work,
                field=field,
                topic_name=topic_name,
            )
            rows.append(row)

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    fieldnames = [
        "field",
        "topic",
        "id",
        "title",
        "abstract",
        "year",
        "type",
        "citation_count",
        "venue",
        "institution",
        "author",
        "country",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Successfully saved {len(rows)} papers to {output_csv}")
    return output_csv


def eda(
    input_csv: str | Path = "../out/openalex/papers.csv",
    null_summary_csv: str | Path = "../out/openalex/papers_null_summary.csv",
    plots_dir: str | Path = "../out/openalex/plots",
    top_n_countries: int = 20,
) -> dict[str, Path]:

    input_csv = Path(input_csv)
    null_summary_csv = Path(null_summary_csv)
    plots_dir = Path(plots_dir)

    null_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    null_like = {"", " ", "nan", "NaN", "None", "null", "NULL", "N/A", "n/a"}
    df_clean = df.copy()

    for col in df_clean.columns:
        df_clean[col] = df_clean[col].apply(
            lambda x: pd.NA
            if pd.isna(x) or (isinstance(x, str) and x.strip() in null_like)
            else x
        )

    # NULL proportion summary
    null_summary = pd.DataFrame({
        "column": df_clean.columns,
        "null_proportion": [
            df_clean[col].isna().mean() for col in df_clean.columns
        ],
    })

    null_summary.to_csv(null_summary_csv, index=False)

    # -------- YEAR PLOT --------
    year_plot = plots_dir / "year_distribution.png"

    if "year" in df_clean.columns:
        year_series = (
            pd.to_numeric(df_clean["year"], errors="coerce").dropna().astype(int)
        )

        if not year_series.empty:
            min_year = year_series.min()
            max_year = year_series.max()

            counts = (
                year_series.value_counts()
                .sort_index()
                .reindex(range(min_year, max_year + 1), fill_value=0)
            )

            plt.figure(figsize=(14, 6))
            ax = counts.plot(kind="bar")

            positions = range(len(counts))

            labels = [
                str(year) if year % 10 == 0 else "" for year in counts.index
            ]

            ax.set_xticks(list(positions))
            ax.set_xticklabels(labels, rotation=45, ha="right")

            plt.xlabel("Year")
            plt.ylabel("Number of Papers")
            plt.title("Papers by Year")
            plt.tight_layout()
            plt.savefig(year_plot, dpi=200)
            plt.close()

    # -------- TYPE PLOT --------
    type_plot = plots_dir / "type_distribution.png"

    if "type" in df_clean.columns:
        type_series = df_clean["type"].dropna().astype(str).str.strip()
        type_series = type_series[type_series != ""]

        if not type_series.empty:
            counts = type_series.value_counts()

            plt.figure(figsize=(10, 6))
            counts.plot(kind="bar")
            plt.xlabel("Type")
            plt.ylabel("Number of Papers")
            plt.title("Papers by Type")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(type_plot, dpi=200)
            plt.close()

    # -------- CITATION COUNT --------
    citation_plot = plots_dir / "citation_count_distribution.png"

    if "citation_count" in df_clean.columns:
        citation_series = pd.to_numeric(
            df_clean["citation_count"], errors="coerce"
        ).dropna()

        if not citation_series.empty:
            max_val = int(citation_series.max())

            bins = np.arange(0, max_val + 10, 10)

            plt.figure(figsize=(12, 6))

            counts, edges, patches = plt.hist(citation_series, bins=bins)

            plt.xlim(left=0)

            ticks = np.arange(0, max_val + 200, 200)
            plt.xticks(ticks, rotation=45)

            for count, patch in zip(counts, patches):
                if count > 0:
                    x = patch.get_x() + patch.get_width() / 2
                    y = count
                    plt.plot(x, y, marker="x")

            plt.xlabel("Citation Count")
            plt.ylabel("Number of Papers")
            plt.title("Citation Count Distribution")
            plt.tight_layout()
            plt.savefig(citation_plot, dpi=200)
            plt.close()

    # -------- COUNTRY PLOT --------
    country_plot = plots_dir / "country_distribution.png"

    if "country" in df_clean.columns:
        countries = (
            df_clean["country"]
            .dropna()
            .astype(str)
            .str.split("|", regex=False)
            .explode()
            .str.strip()
        )

        countries = countries[(countries != "") & (~countries.isna())]

        if not countries.empty:
            counts = countries.value_counts().head(top_n_countries)

            plt.figure(figsize=(12, 6))
            counts.plot(kind="bar")
            plt.xlabel("Country")
            plt.ylabel("Number of Papers")
            plt.title(f"Top {top_n_countries} Countries")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(country_plot, dpi=200)
            plt.close()

    return {
        "null_summary_csv": null_summary_csv,
        "year_plot": year_plot,
        "type_plot": type_plot,
        "citation_plot": citation_plot,
        "country_plot": country_plot,
    }