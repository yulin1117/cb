import os
import json
import re
import sys
import time
import traceback
import requests
from typing import Any

# 💡 Direct imports from your existing project structures to guarantee single-instance reuse
from llm.llm_api import LLM
from llm.prompts import Prompt
class SimpleScadsLLM:
    """
    自建的 ScaDS LLM API 呼叫器
    直接使用 requests 呼叫 ScaDS AI 相容於 OpenAI 規格的 API 端點
    """
    def __init__(self, model: str = "moonshotai/Kimi-K2.7-Code", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.api_url = "https://llm.scads.ai/v1/chat/completions"
        # 優先從環境變數取得 SCADSAI_API_KEY
        KEY_PATH="/home/yuzh952h/workspaces/horse/yuzh952h-cb/CitationBias/api_keys/scads_llm.txt"
        self.api_key = open(KEY_PATH, "r").read().strip()


    def request(self, prompt_text: str) -> str:
        """發送請求至 ScaDS AI LLM 伺服器並回傳文字內容"""
        if not self.api_key:
            raise ValueError("找不到 SCADSAI_API_KEY 環境變數，請確認是否已正確設定環境變數。")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
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
        
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        res_data = response.json()
        return res_data["choices"][0]["message"]["content"]

class SemanticMatchVerifier:
    """
    Quick verification agent using the direct Scads LLM API connector.
    Focuses on auditing Title-Abstract semantic matching with robust JSON parsing,
    reason extraction, and detailed logging.
    """
    def __init__(self, model: str = "moonshotai/Kimi-K2.7-Code", temperature: float = 0.0):
        # 設為 0.0 確保評分邊界極度穩定，不會有隨機性
        self.llm = SimpleScadsLLM(model=model, temperature=temperature)
        self.debug_count = 0  # 用於控制 debug 輸出數量，防止終端機被洗版

    def verify_paper(self, title: str, abstract: str) -> Tuple[int, str]:
        """
        Sends title and abstract to Scads API and forces a clean SCORE and REASON format.
        Includes retry logic and robust key-value / JSON parsing.
        Returns:
            Tuple[int, str]: (score (0 or 1), reason_text)
        """
        # 💡 乾淨且不含大括號的 Prompt，徹底避免伺服器端 template 格式解析錯誤
        user_content = (
            f"You are an academic quality control auditor. Your sole task is to verify if the "
            f"Paper Title and its Abstract match each other semantically.\n\n"
            f"### CRITERIA:\n"
            f"- Score 1: The title and abstract are highly related and match semantically.\n"
            f"- Score 0: The abstract contains web noise, login prompts, redirection artifacts, paywall text, or does not match the title at all.\n\n"
            f"### DATA:\n"
            f"- Title: {title}\n"
            f"- Abstract: {abstract}\n\n"
            f"### REQUIRED OUTPUT FORMAT:\n"
            f"You must return your evaluation in exactly the following format (do not add any preamble, conversational introduction, markdown formatting, or other notes):\n\n"
            f"SCORE: <1 or 0>\n"
            f"REASON: <a concise 1-sentence explanation of your evaluation>"
        )

        answer = ""
        max_retries = 3
        backoff = 1.0

        # 🔄 API 呼叫重試與指數退避邏輯
        for attempt in range(max_retries):
            try:
                # 💡 直接呼叫自建 API 連接器，免除原本複雜的 EmptyPrompt 物件
                raw_res = self.llm.request(user_content)
                if raw_res and raw_res.strip():
                    answer = raw_res
                    break
                else:
                    print(f"⚠️ [API WARNING] Empty response received on attempt {attempt + 1}/{max_retries}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
            except Exception as e:
                print(f"⚠️ [API WARNING] Request failed on attempt {attempt + 1}/{max_retries}: {e}. Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
        else:
            print(f"❌ [API RUNTIME ERROR] All {max_retries} attempts failed or returned empty for paper: {title[:40]}")
            return (0, "API Error: All retries exhausted or empty response received.")

        # 🧪 偵錯點 A：列印前幾篇論文的原始 LLM 回傳，以便肉眼比對格式
        if self.debug_count < 3:
            print(f"\n[DEBUG - Raw LLM Response #{self.debug_count + 1}]")
            print(f"Title: {title[:60]}...")
            print(f"Raw Output Received:\n{answer}")
            print("-" * 60)
            self.debug_count += 1

        # 清洗回傳文字，移除奇怪的空白與引號字元
        clean_answer = answer.strip()
        clean_answer = clean_answer.replace('\xa0', ' ').replace('\u200b', '')
        clean_answer = re.sub(r'[\u201c\u201d\u2018\u2019]', '"', clean_answer)

        # 🎯 解析預設的 SCORE / REASON 鍵值格式
        score = 0
        reason = "Failed to extract reason from LLM output."

        # 1. 嘗試正則解析 SCORE: <0 or 1>
        score_match = re.search(r'SCORE:\s*([01])', clean_answer, re.IGNORECASE)
        if score_match:
            score = int(score_match.group(1))

        # 2. 嘗試正則解析 REASON: <text>
        reason_match = re.search(r'REASON:\s*(.*)', clean_answer, re.IGNORECASE)
        if reason_match:
            reason = reason_match.group(1).strip()
        else:
            # 備用：如果沒有 REASON 標籤，但有 SCORE，拿整個回傳內容做為 Reason 的前 150 字
            if score_match:
                reason = clean_answer.replace('\n', ' ')[:150]

        # 3. 備用：如果沒匹配到 SCORE，嘗試標準 JSON/大括號解析（相容舊版或 LLM 自作聰明回傳 JSON 的狀況）
        if not score_match:
            json_str = ""
            json_match = re.search(r'```json\s*(.*?)\s*```', clean_answer, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                general_match = re.search(r'```\s*(.*?)\s*```', clean_answer, re.DOTALL)
                if general_match:
                    json_str = general_match.group(1).strip()
                else:
                    braces_match = re.search(r'(\{.*\})', clean_answer, re.DOTALL)
                    if braces_match:
                        json_str = braces_match.group(1).strip()

            if json_str:
                try:
                    res_dict = json.loads(json_str)
                    score = int(res_dict.get("score", 0))
                    reason = res_dict.get("reason", "Parsed successfully via JSON backup.")
                    # 限制 score 只能是 0 或 1
                    score = 1 if score == 1 else 0
                    return (score, reason)
                except json.JSONDecodeError as je:
                    if self.debug_count <= 5:
                        print(f"⚠️ [DEBUG - JSON Parse Error]: Failed to parse fallback JSON block: '{json_str[:100]}...' | Error: {je}")

            # 4. 最寬容的純文字特徵碰撞 (極端備用)
            if re.search(r'["\']score["\']\s*:\s*1', clean_answer.lower()) or '"score":1' in clean_answer.lower().replace(" ", ""):
                score = 1
                reason = "Extracted 1 via fallback pattern match."

        return (score, reason)

def run_llm_final_check(topic_url: str, selected_cluster: int, representative_abstracts: int = 50,
                        model: str = "moonshotai/Kimi-K2.7-Code", llm_instance: Any = None,
                        out_root: str = os.path.join("../out", "topics")) -> tuple[int, list[dict]]:
    """
    LLM Final Check block acting as a dynamic pipeline gate filter.
    Reads representatives_top{N}.json, filters the selected_cluster, and counts valid papers.
    Uses the project's native LLM.request format which returns (answer, citations).
    """
    # -----------------------------
    # 1. Resolve paths & target JSON
    # -----------------------------
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
        llm_instance = LLM(model=model, temperature=0.0)
    verified_papers_list=[]
    gold_passed_count = 0
    print(f"     [LLM Final Gate] Auditing {len(papers)} papers from selected Cluster {selected_cluster}...")

    # Create an empty Prompt configuration object natively aligned with your class expectations
    # ─── 💡 核心修正：手動宣告 Mock 物件，徹底繞過實體 Prompt 的基類硬性限制 ───
    class CustomEmptyPrompt:
        filename = "empty_gate.txt" 
        def __init__(self):
            self.prompt = ""   
            self.context = False     

    empty_prompt = CustomEmptyPrompt()

    for idx, p in enumerate(papers, 1):
        title = p.get("title", "") 
        abstract =  p.get("abstract", "")

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


        try:
            # 🎯 Calling the project's native .request() interface.
            answer, _ = llm_instance.request(
                prompt=empty_prompt,
                topic=user_content,
                use_metadata=False,
                context_docs=None
            )
            
            raw_text = answer.strip()
            clean_answer = raw_text.lower()
            
            score = -1
            explanation = "Failed to parse explanation."
            parsed_successfully = False

            # 🧪 Level 1: Attempt direct JSON load on raw text
            try:
                json_data = json.loads(raw_text)
                parsed_successfully = True  
                
                score_val = json_data.get("score")
                if score_val == 1 or score_val is True or str(score_val).strip() == "1":
                    score = 1
                else:
                    score = 0
                    
                explanation = json_data.get("explanation") or json_data.get("reason") or "Direct parsed."

            except (json.JSONDecodeError, Exception):
                parsed_successfully = False 
                print(f"      [DEBUG] Direct JSON parse failed for paper {idx}: {title[:40]}... | Raw: {raw_text}...")

            if score == 1:
                gold_passed_count += 1
                verified_papers_list.append({
                    "id": p.get("id", ""),
                    "title": title,
                    "abstract": abstract,
                    "text": p.get("text", "")
                })

        except Exception as api_err:
            print(f"      ⚠️ Error validating paper {p.get('id', '')[:20]}: {api_err}")
            continue

    return gold_passed_count, verified_papers_list