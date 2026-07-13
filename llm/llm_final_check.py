import os
import json
import re
import sys
import time
import traceback
import requests
from typing import Any, Tuple, List, Dict

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
        
        # Configure payload to suppress thinking token waste on supported models (e.g., Kimi, Qwen, DeepSeek)
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
            # Bubble up standard exceptions for retry logic implementation upstream
            raise RuntimeError(f"Direct API call failed: {e}")

def run_llm_final_check(topic_url: str, selected_cluster: int, representative_abstracts: int = 50,
                        model: str = "moonshotai/Kimi-K2.7-Code", llm_instance: Any = None,
                        out_root: str = os.path.join("../out", "topics")) -> tuple[int, list[dict]]:
    """
    LLM Final Check block acting as a pipeline gate filter.
    Reads representatives_top{N}.json, directly extracts clean title/abstract fields,
    and uses a highly robust parser to evaluate JSON schema containing 'score' and 'explanation'.
    """
    topic_url = topic_url.rstrip("/")
    topic_id: str = topic_url.split("/")[-1]
    save_dir: str = os.path.join(out_root, topic_id)

    # Automatically align with the current resampling budget filename (50 -> 70 -> 90 ...)
    reps_path: str = os.path.join(save_dir, f"representatives_top{representative_abstracts}.json")
    if not os.path.exists(reps_path):
        raise FileNotFoundError(f"Missing representatives file: {reps_path}")

    with open(reps_path, "r", encoding="utf-8") as f:
        reps_data = json.load(f)

    # Isolate the target cluster chosen by the TopicGPT selector
    clusters_obj = reps_data.get("clusters", {})
    target_cluster_data = clusters_obj.get(str(selected_cluster)) or clusters_obj.get(int(selected_cluster))
    
    if not target_cluster_data or not isinstance(target_cluster_data, dict):
        print(f"  [WARNING] Selected cluster {selected_cluster} data not found in JSON.")
        return 0, []

    papers = target_cluster_data.get("papers", [])
    if not papers:
        print(f"  [WARNING] No papers available in cluster {selected_cluster}.")
        return 0, []

    # Re-use the existing LLM instance if passed from upstream, else lazy-initialize a new one
    if llm_instance is None:
        # Set temperature to 0.0 for stable decisions, Kimi will auto-disable thinking in self.extra_body
        llm_instance = SimpleScadsLLM(model=model, temperature=0.0)

    verified_papers_list = []
    gold_passed_count = 0
    print(f"     [LLM Final Gate] Auditing {len(papers)} papers from selected Cluster {selected_cluster}...")

    for idx, p in enumerate(papers, 1):
        # Extract separated clean fields stored by your upgraded cluster_topic
        title = str(p.get("title", "")).strip()
        abstract = str(p.get("abstract", "")).strip()
        
        # Safe fallback: if JSON has not been updated yet, fall back to split strategy
        if not title and not abstract:
            combined_text = str(p.get("text", "")).strip()
            parts = combined_text.split(". ", 1)
            title = parts[0] + "." if len(parts) == 2 else combined_text[:100]
            abstract = parts[1] if len(parts) == 2 else combined_text

        # User content containing your updated prompt instructing 'score' and 'explanation'
        user_content = (
            f"You are an academic quality control auditor. Your sole task is to verify if the "
            f"Paper Title and its Abstract match each other semantically.\n\n"
            f"### CRITERIA:\n"
            f"- Score as 1 if the title and abstract are highly related and match semantically.\n"
            f"- Score as 0 if the abstract contains web noise, login prompts, redirection artifacts, paywall text, or is completely unrelated to the title.\n\n"
            f"### DATA:\n"
            f"- Title: {title}\n"
            f"- Abstract: {abstract}\n\n"
            f"### REQUIRED OUTPUT FORMAT:\n"
            f"You must return a valid JSON object containing exactly the keys 'score' and 'explanation'.\n"
            f"Please follow this exact schema:\n"
            f"{{\n"
            f"  \"score\": 1,\n"
            f"  \"explanation\": \"Provide a one-sentence summary explaining why the title and abstract match or mismatch.\"\n"
            f"}}\n"
            f"Do not add any preamble, conversational introduction, or side notes outside the JSON block."
        )

        answer = ""
        max_retries = 3
        backoff = 1.0

        # API Call Retry & Exponential Backoff Loop
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
            # Fallback block when API retries are completely exhausted
            continue

        raw_text = answer.strip()
        clean_answer = raw_text.lower()
        
        score = 0
        explanation = "Failed to parse explanation."
        parsed_successfully = False

        # 🧪 Level 1: Attempt direct JSON load on raw text
        try:
            json_data = json.loads(raw_text)
            score_val = json_data.get("score")
            if score_val == 1 or score_val is True or str(score_val).strip() == "1":
                score = 1
            explanation = json_data.get("explanation") or json_data.get("reason") or "Direct parsed."
            parsed_successfully = True
            # 🎯 成功時印出提示
            print(f"🎯 [PARSER SUCCESS] Level 1 (Direct JSON Load) succeeded for paper [{idx:02d}]")
        except json.JSONDecodeError:
            # 🔍 失敗時印出原始文字方便 Debug
            print(f"❌ [Level 1 Failed] Paper: {title[:30]}... | Raw snippet: {raw_text[:80]}...")
            pass

        # 🧪 Level 2: Attempt matching Markdown fenced code block
        if not parsed_successfully:
            json_match = re.search(r"```json(?:_topic)?\s*\n(.*?)\n```", raw_text, re.DOTALL | re.IGNORECASE)
            if not json_match:
                json_match = re.search(r"```\s*\n(.*?)\n```", raw_text, re.DOTALL | re.IGNORECASE)
            
            if json_match:
                try:
                    json_data = json.loads(json_match.group(1).strip())
                    score_val = json_data.get("score")
                    if score_val == 1 or score_val is True or str(score_val).strip() == "1":
                        score = 1
                    explanation = json_data.get("explanation") or json_data.get("reason") or "Fenced code parsed."
                    parsed_successfully = True
                    # 🎯 將 print 移到變數成功賦值之後
                    print(f"🎯 [PARSER SUCCESS] Level 2 (Markdown Fenced Code Block) succeeded for paper [{idx:02d}]")
                except json.JSONDecodeError:
                    print(f"⚠️ [Level 2 Match Found But Invalid JSON] Content: {json_match.group(1)[:50]}...")
                    pass

        # 🧪 Level 3: Attempt matching raw outermost curly braces
        if not parsed_successfully:
            brace_match = re.search(r"(\{.*?\})", raw_text, re.DOTALL)
            if brace_match:
                try:
                    json_data = json.loads(brace_match.group(1).strip())
                    score_val = json_data.get("score")
                    if score_val == 1 or score_val is True or str(score_val).strip() == "1":
                        score = 1
                    explanation = json_data.get("explanation") or json_data.get("reason") or "Raw curly parsed."
                    parsed_successfully = True
                    # 🎯 將 print 移到變數成功賦值之後
                    print(f"🎯 [PARSER SUCCESS] Level 3 (Raw Outermost Curly Braces) succeeded for paper [{idx:02d}]")
                except json.JSONDecodeError:
                    print(f"⚠️ [Level 3 Match Found But Invalid JSON] Content: {brace_match.group(1)[:50]}...")
                    pass

        # 🧪 Level 4: Conversational Fallback (Fallback if Llama returns unstructured text)
        if not parsed_successfully:
            # Parse Score via String Collisions
            if '"score": 1' in clean_answer or '"score":1' in clean_answer or "'score': 1" in clean_answer or "'score':1" in clean_answer:
                score = 1
            elif re.search(r'["\']score["\']\s*:\s*1', clean_answer):
                score = 1
            else:
                # NLP Sentiment Fallback
                positive_patterns = ["highly related and match", "match semantically", "is consistent with", "strong connection", "aligns with", "accurate reflection"]
                negative_patterns = ["do not match", "does not match", "unrelated", "web noise", "login prompt", "paywall"]
                has_pos = any(pat in clean_answer for pat in positive_patterns)
                has_neg = any(pat in clean_answer for pat in negative_patterns)
                if has_pos and not has_neg:
                    score = 1

            # Parse Explanation via Regex Extraction
            exp_match = re.search(r'["\']explanation["\']\s*:\s*["\'](.*?)["\']', raw_text, re.IGNORECASE)
            if not exp_match:
                exp_match = re.search(r'["\']reason["\']\s*:\s*["\'](.*?)["\']', raw_text, re.IGNORECASE)
            
            if exp_match:
                explanation = exp_match.group(1)
            else:
                # Clean conversational response text as a fallback explanation
                cleaned_reason = re.sub(r'[{}\"\']', '', raw_text).replace('\n', ' ').strip()
                explanation = f"[NLP Fallback] {cleaned_reason[:150]}..."
            
            # 🎯 標記 Level 4 成功觸發
            print(f"🚨 [PARSER FALLBACK] Level 4 (NLP String Fallback) used for paper [{idx:02d}]. Score forced to {score}")


        if score == 1:
            gold_passed_count += 1
            verified_papers_list.append({
                "id": p.get("id", ""),
                "title": title,
                "abstract": abstract,
                "text": p.get("text", ""),
                "llm_score":score,
                "llm_explanation":explanation
            })

    return gold_passed_count, verified_papers_list