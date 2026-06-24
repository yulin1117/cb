import os
from collections import Counter
import csv 
import json
import re
from test2 import fetch_unfiltered_raw_works,reconstruct_abstract  # 從 test2.py 中匯入原始資料撈取函式
# 嘗試安全匯入 vLLM 與 transformers 相關套件
try:
    from vllm import LLM as VLLMEngine, SamplingParams
    from transformers import AutoTokenizer
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# 嘗試安全匯入 OpenAI 串接套件
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────
# 全域組態設定
# ──────────────────────────────────────────────────────────────────────
BASE_URL = "MODEL_URL"  # 指定您的模型網址
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
VLLM_MODEL_NAME = "Qwen/Qwen3.5-9B"  # 本次使用的 Qwen3.5-9B 模型
VLLM_TENSOR_PARALLEL = 1
VLLM_BATCH_SIZE = 64  # 每次發送到 vLLM 的提示詞批次大小

# 延遲初始化變數
_vllm_engine = None
_vllm_tokenizer = None

def _write_audit_csv(data_list: list[dict], csv_filename: str = "data_cleaning_audit_report.csv"):
    """
    將所有論文的資料清洗與稽核結果（包含 APPROVED 與 REJECTED）安全寫入指定的 .csv 檔案。
    
    參數:
    - data_list: 包含所有論文字典的列表 (包含 llm_processed_papers 與 hard_rejected_papers)
    - csv_filename: 輸出的 CSV 檔案名稱，預設為 'data_cleaning_audit_report.csv'
    """
    if not data_list:
        print("⚠️ [CSV WARNING] 沒有收到任何論文資料，取消寫入 CSV。")
        return

    # 定義 CSV 報告的標準表頭 (Columns)
    csv_fields = ["id", "title", "decision", "is_english", "is_matching", "audit_reason"]
    
    try:
        print(f"\n💾 正在將 {len(data_list)} 篇論文的稽核結果寫入 {csv_filename} ...")
        
        # 使用 utf-8 確保標題與原因中的各國語系、雜訊符號不會亂碼
        # newline="" 可以防止 Windows 系統下每行之間出現多餘的空行
        with open(csv_filename, mode="w", encoding="utf-8", newline="") as f:
            # extrasaction="ignore" 可以防止字典裡有其他 OpenAlex 原始欄位時導致噴錯
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            
            # 1. 寫入第一行的欄位名稱 (Header)
            writer.writeheader()
            
            # 2. 批次寫入所有論文的清洗紀錄
            writer.writerows(data_list)
            
        print(f"🎉 審計 CSV 報告導出成功！")
        print(f"📍 檔案絕對路徑: {os.path.abspath(csv_filename)}")
        
    except Exception as csv_err:
        print(f"❌ [CSV ERROR] 寫入 CSV 檔案時發生異常錯誤: {csv_err}")

def filter_missing_metadata(papers: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    工作流 (1)e 規則：檢查缺失元數據（標題、摘要、作者）。
    回傳: (通過名單, 被踢出名單)
    """
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
            
            # 貼上被踢出的原因與決策標籤
            paper["decision"] = "REJECTED"
            paper["is_english"] = False
            paper["is_matching"] = False
            paper["audit_reason"] = f"Missing critical metadata: {missing_fields}"
            rejected.append(paper)
            print(f"⚠️ [METADATA MISSING] 論文 {paper_id} 遺失 {missing_fields} -> 直接剔除")
        else:
            passed.append(paper)
            
    return passed, rejected


def filter_duplicate_abstract(papers: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    工作流 (1)d 規則：全域摘要去重碰撞攔截（碰撞懲罰機制）。
    回傳: (通過名單, 被踢出名單)
    """
    passed = []
    rejected = []
    
    # 統計此階段所有論文的摘要頻次（排除空字串）
    abstract_counter = Counter([p["reconstructed_abstract"] for p in papers if p.get("reconstructed_abstract")])
    
    for paper in papers:
        paper_id = paper.get("id", "").split("/")[-1] or "Unknown"
        abstract_text = paper.get("reconstructed_abstract", "")
        
        # 如果該摘要在整批資料中出現 2 次以上，觸發碰撞懲罰
        if abstract_text and abstract_counter[abstract_text] >= 2:
            paper["decision"] = "REJECTED"
            paper["is_english"] = False
            paper["is_matching"] = False
            paper["audit_reason"] = f"Duplicate Abstract Collision: This abstract appears {abstract_counter[abstract_text]} times across the dataset."
            rejected.append(paper)
            print(f"🚨 [DUPLICATE COLLISION] 論文 {paper_id} 觸發摘要重複碰撞 ({abstract_counter[abstract_text]} 次) -> 直接剔除")
        else:
            passed.append(paper)
            
    return passed, rejected


def filter_by_hard_rules(papers: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    硬性規則總閘門：串聯「缺失檢查」與「重複去重」兩個濾網。
    回傳: (最終通過可送 LLM 的名單, 所有被硬性規則淘汰的名單總和)
    """
    print("\n[Phase 1] 開始執行硬性規則防火牆 (Hard Rules Pipeline)...")
    
    # 1. 先將 OpenAlex 倒排索引還原為純文字並標準化
    for paper in papers:
        inv_index = paper.get("abstract_inverted_index")
        abstract_text = reconstruct_abstract(inv_index) if inv_index else ""
        paper["reconstructed_abstract"] = " ".join(abstract_text.strip().split())

    # 2. 第一道濾網：踢出缺失元數據
    meta_passed, meta_rejected = filter_missing_metadata(papers)
    
    # 3. 第二道濾網：針對過篩後的論文，再踢出摘要重複碰撞
    final_passed, dup_rejected = filter_duplicate_abstract(meta_passed)
    
    # 合併所有被規則踢出的論文
    all_hard_rejected = meta_rejected + dup_rejected
    
    print(f"📋 Phase 1 結束：原始 {len(papers)} 篇 | 規則剔除 {len(all_hard_rejected)} 篇 | 剩餘 {len(final_passed)} 篇可送 LLM")
    return final_passed, all_hard_rejected

def _load_api_key() -> str:
    """
    從系統環境變數中安全讀取 OpenAI 協定的 API 金鑰
    """
    return os.environ.get("OPENAI_API_KEY", "")

def _get_vllm_engine():
    """
    延遲且安全地初始化 vLLM 引擎與分詞器。
    手動設定 Triton 後端以繞過特定環境（如 CUDA 12.4 與 SM90）下的編譯與快取寫入錯誤。
    """
    global _vllm_engine, _vllm_tokenizer
    if _vllm_engine is None:
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM 或 transformers 模組未安裝！請先在您的環境中安裝。")
        
        print(f"正在加載分詞器: {VLLM_MODEL_NAME} ...")
        _vllm_tokenizer = AutoTokenizer.from_pretrained(VLLM_MODEL_NAME)
        
        print(f"正在初始化 vLLM 引擎: {VLLM_MODEL_NAME} (並強制回退至 Triton 後端) ...")
        _vllm_engine = VLLMEngine(
            model=VLLM_MODEL_NAME,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL,
            # 強制回退到 Triton，避開 CUDA 與 SM90 的編譯或權限錯誤
            gdn_prefill_backend="triton", 
            # 確保注意力機制模組使用相容後端
            attention_backend="TRITON_ATTN",
            max_num_seqs=512,
            max_model_len=4096, 
            trust_remote_code=True,
            #enforce_eager=True  # 強制啟用 eager 模式以提升兼容性與穩定性
        )
    return _vllm_engine, _vllm_tokenizer


def _format_prompt(system_prompt: str, user_content: str) -> str:
    """
    格式化訊息為 Qwen 支援的 Chat Template 格式
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return _vllm_tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=False
    )

# 建立抽樣參數 (Sampling Parameters)
if VLLM_AVAILABLE:
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=256,
        stop=["<|im_end|>", "<|endoftext|>"]
    )
else:
    sampling_params = None

# ──────────────────────────────────────────────────────────────────────
# LLM 篩選提示詞內容 (Prompt definition)
# ──────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert academic data cleaning assistant. "
    "Your task is to identify two key data quality aspects of an academic paper: "
    "1. If the language of both the title and abstract is primarily English. "
    "2. If the semantic content of the title and the abstract are highly relevant and match each other. "
    "Note that the data may contain irrelevant noise such as duplicate abstracts, misplacements, "
    "paywall indicators, login prompts, or redirect text. "
    "You must analyze the input and output true/false in a strict, valid JSON format with NO other conversational text."
)


def get_user_content(title: str, abstract: str) -> str:
    """
    動態組裝待評估的學術論文使用者提示詞
    """
    return f"""### EVALUATION CRITERIA:
1. LANGUAGE CHECK: Is the text written in English? Output false if it is in other languages.
2. MATCHING CHECK: Do the Title and Abstract match semantically? Output false if the abstract is unrelated content or noise (e.g., "This content is only available as a PDF", "You do not currently have access", "Share on Facebook").

### INPUT DATA:
- Paper Title: {title}
- Paper Abstract: {abstract}

### OUTPUT FORMAT:
Respond ONLY with a valid JSON object:
{{
  "is_english": true/false,
  "is_matching": true/false,
  "reason": "A brief reason"
}}"""


def _generate_single(system_prompt: str, user_content: str, temperature: float, backend: str) -> str:
    """
    單次推理介面，支援 vLLM 本地推理引擎或透過 OpenAI API 遠端調用
    """
    if backend == "vllm":
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed. Install it with: pip install vllm")
        engine, tokenizer = _get_vllm_engine()
        sampling_params = SamplingParams(temperature=temperature, max_tokens=256, stop=["<|im_end|>", "<|endoftext|>"])
        prompt = _format_prompt(system_prompt, user_content)
        outputs = engine.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()
    else:
        if not OPENAI_AVAILABLE:
            raise RuntimeError("openai is not installed. Install it with: pip install openai")
        my_api_key = _load_api_key()
        client = OpenAI(base_url=BASE_URL, api_key=my_api_key)
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            model=MODEL_NAME,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
def batch_filter_papers(papers: list[dict], backend: str = "vllm") -> list[dict]:
    if not papers:
        return []

    # ─── 核心整合：直接呼叫剛剛做好的硬性規則大閘門 ───
    valid_papers_for_llm, hard_rejected_papers = filter_by_hard_rules(papers)

    # 如果規則把論文全殺光了，直接提早寫入 CSV 並結束
    if not valid_papers_for_llm:
        _write_audit_csv(hard_rejected_papers, "data_cleaning_audit_report.csv")
        return []

    # ─── [Phase 2] LLM 語意精洗核心 (vLLM / API 推理) ───
    raw_outputs = []
    if backend == "vllm":
        if not VLLM_AVAILABLE:
            print("⚠️ [錯誤] vLLM 未成功匯入，無法執行 vLLM 批次推理過濾！")
            return []
        engine, tokenizer = _get_vllm_engine()
        prompts = []
        for paper in valid_papers_for_llm:
            user_content = get_user_content(paper["title"], paper["reconstructed_abstract"])
            prompt = _format_prompt(SYSTEM_PROMPT, user_content)
            prompts.append(prompt)
        
        print(f"\n🚀 vLLM 開始批次語意稽核 {len(prompts)} 篇論文...")
        temp_sampling_params = SamplingParams(temperature=0.0, max_tokens=256, stop=["<|im_end|>", "<|endoftext|>"])
        outputs = engine.generate(prompts, temp_sampling_params)
        raw_outputs = [out.outputs[0].text.strip() for out in outputs]
    else:
        # API 模式（此處省略，保持與先前相同）
        pass

    # ─── [Phase 3] 解析 LLM 結果與最終決策 ───
    approved_papers = []
    llm_processed_papers = []
    
    for i, raw_text in enumerate(raw_outputs):
        orig_paper = valid_papers_for_llm[i]
        paper_id = orig_paper.get("id", "").split("/")[-1] or "Unknown"
        
        try:
            llm_result = json.loads(raw_text)
            is_english = llm_result.get("is_english")
            is_matching = llm_result.get("is_matching")
            reason = llm_result.get("reason", "No reason provided.")
            
            orig_paper["is_english"] = is_english
            orig_paper["is_matching"] = is_matching
            
            if is_english is True and is_matching is True:
                orig_paper["decision"] = "APPROVED"
                orig_paper["audit_reason"] = reason
                approved_papers.append(orig_paper)
                print(f"✅ [APPROVED] 論文: {paper_id} | 原因: {reason}")
            else:
                orig_paper["decision"] = "REJECTED"
                orig_paper["audit_reason"] = reason
                print(f"❌ [REJECTED] 論文: {paper_id} | LLM理由: {reason}")
                
        except Exception as e:
            orig_paper["decision"] = "REJECTED"
            orig_paper["audit_reason"] = f"LLM json parse error: {str(e)}"
            
        llm_processed_papers.append(orig_paper)

    # ─── 💾 最終合流並導出至 CSV ───
    # 這裡包含了：LLM 處理的論文 + 一開始就被硬性規則（缺失或重複）踢出的所有論文
    final_csv_list = llm_processed_papers + hard_rejected_papers
    _write_audit_csv(final_csv_list, "data_cleaning_audit_report2.csv")

    return approved_papers
if __name__ == "__main__":
    raw_unfiltered_papers = fetch_unfiltered_raw_works(topic_id="T14423", num_papers=1000, oa_status="diamond", lang_filter="en")
    
    backend_to_test = None
    if VLLM_AVAILABLE:
        backend_to_test = "vllm"
        print(f"檢測到 vLLM 環境，將使用 vLLM 後端 ({VLLM_MODEL_NAME}) 進行測試...")
    elif OPENAI_AVAILABLE:
        backend_to_test = "api"
        print(f"未檢測到 vLLM，但檢測到 OpenAI 套件已安裝，將使用 API 後端 ({MODEL_NAME}) 進行測試...")
    else:
        print("⚠️ 未檢測到 vLLM 環境或 OpenAI 套件，跳過主程式中的直接測試流程。請確認環境部署。")

    if backend_to_test:
        clean_results = batch_filter_papers(raw_unfiltered_papers, backend="vllm")
        
        print("=" * 60)
        print(f"  ● 清洗通過總數 : {len(clean_results)} 篇")
        print("=" * 60)