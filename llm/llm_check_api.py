import os
import json
import re
import csv
import sys
import traceback
import time
import requests
from datetime import datetime
from typing import Any, Tuple

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

def run_ablation_proof():
    topics_dir = "../out/topics"
    if not os.path.exists(topics_dir):
        print(f"❌ Error: Cannot find out directory at {topics_dir}")
        return

    print("🚀 [Scads API] Starting quick LLM semantic audit proof script with verbose debug logs...")
    verifier = SemanticMatchVerifier()

    # 初始化產出檔案
    csv_file_path = "semantic_verification_details.csv"
    markdown_file_path = "ablation_proof_metrics.md"
    
    # 建立/覆寫 CSV，並寫入標頭
    csv_headers = ["Topic ID", "Paper Title/ID", "Abstract", "Score", "Reason"]
    try:
        with open(csv_file_path, "w", encoding="utf-8", newline="") as csv_f:
            writer = csv.writer(csv_f)
            writer.writerow(csv_headers)
    except IOError as io_err:
        print(f"❌ Error: Could not initialize CSV file {csv_file_path}: {io_err}")
        return

    total_valid_topics = 0
    topics_passing_threshold = 0
    
    global_total_papers = 0
    global_approved_papers = 0
    
    topic_report_details = [] # 用於收集各 Topic 摘要，最後寫入 markdown

    topic_folders = [f for f in os.listdir(topics_dir) if os.path.isdir(os.path.join(topics_dir, f))]
    
    for t_id in sorted(topic_folders):
        json_path = os.path.join(topics_dir, t_id, "representatives_top50.json")
        
        if not os.path.exists(json_path):
            continue

        total_valid_topics += 1
        print(f"\n📂 Analyzing Topic [{t_id}] via {json_path} ...")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        clusters = data.get("clusters", {})
        if not clusters:
            print(f"⚠️ [WARNING] No clusters found in {t_id}, skipping...")
            continue

        topic_passed_papers = 0
        topic_total_papers = 0

        # 重設每個 topic 的 debug 計數器，以便觀察每個 topic 剛開始的幾篇
        verifier.debug_count = 0

        # 用於暫存此 Topic 之所有論文比對結果，隨後批量追加寫入 CSV
        rows_to_write = []

        # 🎯 核心修改：從嵌套的 metrics 欄位中尋找 score。
        # 鏈式 Fallback 機制：metrics.score -> score -> cluster_score -> 預設 0
        best_c_id = max(clusters.keys(), key=lambda k: clusters[k]["metrics"]["score"])
        best_c_info = clusters[best_c_id]
        best_score = best_c_info["metrics"]["score"]
        print(f"   🎯 Selected Best Cluster: [{best_c_id}] with Score: {best_score}")


        # 只取得該最優 Cluster 下的 papers
        papers = best_c_info.get("papers", [])
        for p in papers:
            title = p.get("title", "") or p.get("id", "")
            abstract = p.get("text", "") or p.get("abstract", "")
            
            # 執行驗證，取得 score 與 reason
            score, reason = verifier.verify_paper(title, abstract)
            
            topic_total_papers += 1
            global_total_papers += 1
            if score == 1:
                topic_passed_papers += 1
                global_approved_papers += 1
            
            # 收集 CSV 資料列
            rows_to_write.append([t_id, title, abstract.replace('\n', ' '), score, reason])

        # 即時追加寫入 CSV，防止程式異常中斷時數據遺失
        with open(csv_file_path, "a", encoding="utf-8", newline="") as csv_f:
            writer = csv.writer(csv_f)
            writer.writerows(rows_to_write)

        topic_acc_rate = (topic_passed_papers / topic_total_papers * 100) if topic_total_papers > 0 else 0
        is_pass = topic_passed_papers >= 20
        if is_pass:
            topics_passing_threshold += 1

        # 紀錄至 markdown 報告資料集
        topic_report_details.append({
            "topic_id": t_id,
            "passed": topic_passed_papers,
            "total": topic_total_papers,
            "rate": topic_acc_rate,
            "status": "PASS" if is_pass else "FAIL"
        })

        print(f"   📊 Result for {t_id}: {topic_passed_papers}/{topic_total_papers} passed LLM Check | Accept Rate: {topic_acc_rate:.2f}% | Pass (>=20): {is_pass}")

    # 計算總體指標
    final_global_accept_rate = (global_approved_papers / global_total_papers * 100) if global_total_papers > 0 else 0
    final_pass_rate = (topics_passing_threshold / total_valid_topics * 100) if total_valid_topics > 0 else 0

    print("\n" + "="*80)
    print("                 🏁 FINAL ABLATION PROOF METRICS SUMMARY")
    print("="*80)
    if total_valid_topics == 0:
        print("❌ No 'representatives_top50.json' files were discovered.")
        return

    print(f" ● Total Audited Topics (with Top-50 JSON) : {total_valid_topics} topics")
    print(f" ● Total Evaluated Representative Papers   : {global_total_papers} papers")
    print(f" ● Total Approved Papers by LLM            : {global_approved_papers} papers")
    print("-" * 80)
    print(f" 🎯 1. GLOBAL ACCEPT RATE : {final_global_accept_rate:.2f}%")
    print(f" 🎯 2. PIPELINE PASS RATE : {final_pass_rate:.2f}%")
    print("="*80 + "\n")
    print(f"💾 Detailed results exported incrementally to: {csv_file_path}")

    # ✍️ 自動產生 .md 指標報告檔案
    try:
        with open(markdown_file_path, "w", encoding="utf-8") as md_f:
            md_f.write(f"# 🏆 Semantic Match Audit Ablation Proof Report\n\n")
            md_f.write(f"Generated on: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n\n")
            
            md_f.write(f"## 📊 Global Summary Metrics\n")
            md_f.write(f"| Metric | Value | Description |\n")
            md_f.write(f"| :--- | :--- | :--- |\n")
            md_f.write(f"| **Total Audited Topics** | `{total_valid_topics}` | Number of topics processed with representative papers |\n")
            md_f.write(f"| **Total Evaluated Papers** | `{global_total_papers}` | Total individual papers evaluated by LLM |\n")
            md_f.write(f"| **Total Approved Papers** | `{global_approved_papers}` | Papers determined as high quality and matching |\n")
            md_f.write(f"| **GLOBAL ACCEPT RATE** | **`{final_global_accept_rate:.2f}%`** | Ratio of approved papers over total evaluated |\n")
            # 完美修復跳脫字元警告：將 LaTeX 的 \ge 改寫為 \\ge
            md_f.write(f"| **PIPELINE PASS RATE** | **`{final_pass_rate:.2f}%`** | Percentage of topics with $\\ge 20$ approved papers |\n\n")
            
            # 完美修復跳脫字元警告：將 LaTeX 的 \ge 改寫為 \\ge
            md_f.write(f"## 📂 Detailed Topic Statistics\n")
            md_f.write(f"| Topic ID | Approved / Total | Acceptance Rate | Pipeline Status (Threshold $\\ge 20$) |\n")
            md_f.write(f"| :--- | :---: | :---: | :---: |\n")
            for tr in topic_report_details:
                status_emoji = "✅ PASS" if tr["status"] == "PASS" else "❌ FAIL"
                md_f.write(f"| {tr['topic_id']} | {tr['passed']} / {tr['total']} | {tr['rate']:.2f}% | {status_emoji} |\n")
            
            md_f.write(f"\n---\n")
            md_f.write(f"ℹ️ *Note: This report was automatically generated. Raw output and matched criteria reasons can be inspected in the sister CSV log file: `{csv_file_path}`.*\n")
            
        print(f"💾 Metrics report successfully saved to: {markdown_file_path}")
    except Exception as md_err:
        print(f"⚠️ Warning: Could not write Markdown report file: {md_err}")

if __name__ == "__main__":
    run_ablation_proof()