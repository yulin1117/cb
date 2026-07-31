import os
import json
import re
import sys
import time
import traceback
import requests
from typing import Any, Tuple, List, Dict
from tqdm import tqdm
class SimpleScadsLLM:
    """
    Self-contained standalone ScaDS LLM API Client.
    Directly posts HTTP payloads to ScaDS AI endpoint without relying on project-native classes,
    preventing abstract-class violations, circular imports, and token overhead.
    """
    def __init__(self, model: str = "moonshotai/Kimi-K2.7-Code", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.api_url = "https://llm.scads.ai/v1/chat/completions"
        self.api_key = self._load_api_key()

    def _load_api_key(self) -> str:
        """
        Loads ScaDS API key from standard project directories, supporting absolute path fallback.
        """
        paths_to_try = [
            "api_keys/scads_llm.txt",
            "../api_keys/scads_llm.txt",
            "/home/yuzh952h/workspaces/horse/yuzh952h-cb/CitationBias/api_keys/scads_llm.txt"
        ]
        for p in paths_to_try:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        key = f.read().strip()
                        if key:
                            return key
                except Exception:
                    pass
        raise FileNotFoundError("Could not find or read api_keys/scads_llm.txt under any search path.")

    def request(self, prompt_text: str) -> str:
        """
        Sends raw prompt payload to ScaDS AI with disabled reasoning thinking blocks.
        """
        if not self.api_key:
            raise ValueError("ScaDS AI API key is not configured.")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Configure payload to suppress thinking token waste on supported models
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt_text}
            ],
            "temperature": self.temperature,
            "extra_body": {
                "chat_template_kwargs": {
                    "thinking": False
                }
            }
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            res_data = response.json()
            return res_data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Direct API call failed: {e}")

def run_llm_final_check(topic_url: str, selected_cluster: int, representative_abstracts: int = 50,
                        model: str = "moonshotai/Kimi-K2.7-Code", llm_instance: Any = None,
                        out_root: str = os.path.join("../out", "topics")) -> tuple[int, list[dict]]:
    """
    LLM Final Check block acting as a quality gate filter.
    Audits papers based on standard IEEE/Google Scholar publication criteria:
      1. match: Semantic alignment between Title and Abstract
      2. eng  : Language check (title non-eng <= 50%, abstract non-eng <= 20%, proper nouns allowed)
      3. clean: Quality/format check (no HTML tags, broken words, paywalls, redirections)
      4. pass : Evaluated as (match AND eng AND clean)
    """
    topic_url = topic_url.rstrip("/")
    topic_id: str = topic_url.split("/")[-1]
    save_dir: str = os.path.join(out_root, topic_id)
    #load existing cache if available
    target_file = os.path.join(save_dir, "final_verified_papers.json")
    if os.path.exists(target_file):
        print(f"[CACHE-LLM] Topic {topic_id} already has final_verified_papers.json. Parsing...")
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            verified_papers = data.get("papers", [])
            gold_passed_count = data.get("total_verified", len(verified_papers))
            print(f"[CACHE-LLM] Successfully loaded {gold_passed_count} verified papers from cache.")
            # 🎯 完美還原外層所需的 (int, list) 回傳結構
            return gold_passed_count, verified_papers
            
        except Exception as cache_err:
            print(f"⚠️ [CACHE-LLM] Failed to read verification cache ({cache_err}). Re-running LLM Gate...")
            # 發生任何讀取異常，pass 掉讓程式往下走，重新跑 LLM 確保安全
            pass

    reps_path: str = os.path.join(save_dir, f"cluster_papers.json")
    audit_log_path: str = os.path.join(save_dir, "audit_results.json")

    if not os.path.exists(reps_path):
        raise FileNotFoundError(f"Missing representatives file: {reps_path}")

    with open(reps_path, "r", encoding="utf-8") as f:
        reps_data = json.load(f)

    clusters_obj = reps_data.get("clusters", {})
    target_cluster_data = clusters_obj.get(str(selected_cluster)) or clusters_obj.get(int(selected_cluster))
    
    if not target_cluster_data or not isinstance(target_cluster_data, dict):
        print(f"  [WARNING] Selected cluster {selected_cluster} data not found in JSON.")
        return 0, []

    all_cluster_papers = target_cluster_data.get("papers", [])
    papers = all_cluster_papers[:representative_abstracts]
    if not papers:
        print(f"  [WARNING] No papers available in cluster {selected_cluster}.")
        return 0, []

    # 2. 載入已有的快取記錄 (audit_results.json)
    audited_cache: Dict[str, Dict] = {}
    if os.path.exists(audit_log_path):
        try:
            with open(audit_log_path, "r", encoding="utf-8") as f:
                cached_list = json.load(f)
                audited_cache = {item["work_id"]: item for item in cached_list}
            print(f"  📦 Loaded {len(audited_cache)} previously audited records from cache.")
        except Exception as e:
            print(f"  ⚠️ Failed to read audit cache, starting fresh: {e}")

    # 3. 找出需要新增審查的論文 (Unchecked Candidates)
    papers_to_check = []
    for p in papers:
        work_id = p.get("id", p.get("work_url", ""))
        if work_id not in audited_cache:
            papers_to_check.append(p)

    print(f"  🔍 Budget requested: top{representative_abstracts} | New papers to audit via LLM: {len(papers_to_check)}")

    # 4. 僅對未審查的新論文執行 LLM API 呼叫
    if papers_to_check and llm_instance is None:
        llm_instance = SimpleScadsLLM(model=model, temperature=0.0)

    verified_papers_list = []
    gold_passed_count = 0
    print(f"[LLM Final Gate] Auditing {len(papers)} papers from selected Cluster {selected_cluster}...")
    new_audits_performed = False
    for idx, p in enumerate(tqdm(papers, desc="Processing papers"), 1):
        work_id = str(p.get("id","")).strip()
        title = str(p.get("title", "")).strip()
        abstract = str(p.get("abstract", "")).strip()
        
        if not title and not abstract:
            combined_text = str(p.get("text", "")).strip()
            parts = combined_text.split(". ", 1)
            title = parts[0] + "." if len(parts) == 2 else combined_text[:100]
            abstract = parts[1] if len(parts) == 2 else combined_text

        if work_id in audited_cache:
            continue

        new_audits_performed = True
        # Updated Prompt aligned with IEEE / Google Scholar publishing standards
        user_content = (
            f"You are an academic publication auditor for major indexing platforms (e.g., IEEE, Google Scholar).\n"
            f"Evaluate if the given Paper Title and Abstract meet standard publishing criteria across three dimensions:\n\n"
            f"### CRITERIA:\n"
            f"1. `match` (boolean): Does the abstract semantically align with the title? Return false if misplaced or describing an unrelated study.\n"
            f"2. `eng` (boolean): Is the text predominantly English? Minor foreign proper nouns, institution names, or native parenthetical terms are acceptable. Return false if non-English content exceeds 50% in Title, or 20% in Abstract.\n"
            f"3. `clean` (boolean): Is the format clean like a standard publication? Return false if text contains PDF/OCR noise (e.g. broken words like 'com-press', 'ques- tionnaire'), HTML tags (<p>, <i>, &amp;), truncation, paywall notices, or redirection text. (1-2 standard DOI/publisher links at the end are acceptable).\n\n"
            f"### DATA:\n"
            f"- Title: {title}\n"
            f"- Abstract: {abstract}\n\n"
            f"### REQUIRED OUTPUT FORMAT:\n"
            f"Return ONLY a valid JSON object matching this exact schema:\n"
            f"{{\n"
            f'  "match": true,\n'
            f'  "eng": true,\n'
            f'  "clean": true,\n'
            f'  "explanation": "Provide a 1-sentence summary of what the paper is about. If any criteria fail (false), explain specifically why."\n'
            f"}}\n"
            f"Do not include any preamble, conversational text, or side notes outside the JSON block."
        )

        answer = ""
        max_retries = 3
        backoff = 1.0

        for attempt in range(max_retries):
            try:
                raw_res = llm_instance.request(user_content)
                if raw_res and raw_res.strip():
                    answer = raw_res
                    break
                time.sleep(backoff)
                backoff *= 2
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"      ⚠️ API failure on paper {idx}: {e}")
                time.sleep(backoff)
                backoff *= 2
        else:
            continue

        raw_text = answer.strip()
        clean_answer = raw_text.lower()
        
        match_val = False
        eng_val = False
        clean_val = False
        explanation = "Failed to parse explanation."
        parsed_successfully = False

        # 🧪 Level 1: Direct JSON parse
        try:
            json_data = json.loads(raw_text)
            match_val = bool(json_data.get("match", False))
            eng_val = bool(json_data.get("eng", False))
            clean_val = bool(json_data.get("clean", False))
            explanation = str(json_data.get("explanation") or json_data.get("reason") or "Direct parsed.")
            parsed_successfully = True
        except json.JSONDecodeError:
            pass

        # 🧪 Level 2: Markdown fenced code block
        if not parsed_successfully:
            json_match = re.search(r"```json(?:_topic)?\s*\n(.*?)\n```", raw_text, re.DOTALL | re.IGNORECASE)
            if not json_match:
                json_match = re.search(r"```\s*\n(.*?)\n```", raw_text, re.DOTALL | re.IGNORECASE)
            
            if json_match:
                try:
                    json_data = json.loads(json_match.group(1).strip())
                    match_val = bool(json_data.get("match", False))
                    eng_val = bool(json_data.get("eng", False))
                    clean_val = bool(json_data.get("clean", False))
                    explanation = str(json_data.get("explanation") or json_data.get("reason") or "Fenced code parsed.")
                    parsed_successfully = True
                except json.JSONDecodeError:
                    pass

        # 🧪 Level 3: Raw curly braces extraction
        if not parsed_successfully:
            brace_match = re.search(r"(\{.*?\})", raw_text, re.DOTALL)
            if brace_match:
                try:
                    json_data = json.loads(brace_match.group(1).strip())
                    match_val = bool(json_data.get("match", False))
                    eng_val = bool(json_data.get("eng", False))
                    clean_val = bool(json_data.get("clean", False))
                    explanation = str(json_data.get("explanation") or json_data.get("reason") or "Raw curly parsed.")
                    parsed_successfully = True
                except json.JSONDecodeError:
                    pass

        # 🧪 Level 4: Fallback heuristic string inspection
        if not parsed_successfully:
            if '"match": true' in clean_answer or '"match":true' in clean_answer: match_val = True
            if '"eng": true' in clean_answer or '"eng":true' in clean_answer: eng_val = True
            if '"clean": true' in clean_answer or '"clean":true' in clean_answer: clean_val = True
            
            exp_match = re.search(r'["\']explanation["\']\s*:\s*["\'](.*?)["\']', raw_text, re.IGNORECASE)
            if exp_match:
                explanation = exp_match.group(1)
            else:
                cleaned_reason = re.sub(r'[{}\"\']', '', raw_text).replace('\n', ' ').strip()
                explanation = f"[NLP Fallback] {cleaned_reason[:150]}..."

        # Calculate logical pass condition
        pass_val = match_val and eng_val and clean_val

        #print(f"[{idx:02d}] Paper: {title[:35]}... | Match: {match_val} | Eng: {eng_val} | Clean: {clean_val} | Pass: {pass_val}")
        #print(f"Expl: {explanation[:90]}...")
        # Update cache dictionary with new record
        audited_cache[work_id] = {
            "id":work_id,
            "title": title,
            "abstract": abstract,
            "text": p.get("text", ""),
            "match": match_val,
            "eng": eng_val,
            "clean": clean_val,
            "pass": pass_val,
            "llm_explanation": explanation
        }

    if new_audits_performed or not os.path.exists(audit_log_path):
        try:
            with open(audit_log_path, "w", encoding="utf-8") as f:
                json.dump(list(audited_cache.values()), f, ensure_ascii=False, indent=2)
            print(f"  💾 Updated audit cache ({len(audited_cache)} total records) saved to: {audit_log_path}")
        except Exception as e:
            print(f"  ⚠️ Failed to write audit cache to {audit_log_path}: {e}")

    verified_papers_list = []
    gold_passed_count = 0

    for p in papers:
        work_id = str(p.get("id", p.get("work_url", "")))
        record = audited_cache.get(work_id)
        if record and record.get("pass") is True:
            gold_passed_count += 1
            verified_papers_list.append(record)

    return gold_passed_count, verified_papers_list