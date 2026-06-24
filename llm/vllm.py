import sys
import json
import os
try:
    from vllm import LLM as VLLMEngine, SamplingParams
    from transformers import AutoTokenizer
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

VLLM_MODEL_NAME = "Qwen/Qwen3.5-9B" 
VLLM_TENSOR_PARALLEL = 1
_vllm_engine = None
_vllm_tokenizer = None

def _get_vllm_engine():
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
            gdn_prefill_backend="triton", 
            attention_backend="TRITON_ATTN",
            max_num_seqs=512,
            max_model_len=4096, 
            trust_remote_code=True,
        )
    return _vllm_engine, _vllm_tokenizer

def _format_prompt(system_prompt: str, user_content: str) -> str:
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

if VLLM_AVAILABLE:
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=256,
        stop=["<|im_end|>", "<|endoftext|>"]
    )
else:
    sampling_params = None
SYSTEM_PROMPT = (
    ""
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
  "match_score": ...
}}"""

def main():
    if len(sys.argv) < 3:
        print("Usage: python vllm_filter_service.py <input_json> <output_json>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    with open(input_path, "r", encoding="utf-8") as f:
        papers = json.load(f)
        
    from vllm import SamplingParams
    engine, tokenizer = _get_vllm_engine()
    
    prompts = []
    for paper in papers:
        # 取得 title 與 abstract
        title = paper.get("title", "")
        # 相容原本 API payload 的欄位名
        abstract = paper.get("reconstructed_abstract") or paper.get("abstract") or ""
        
        user_content = get_user_content(title, abstract)
        prompt = _format_prompt(SYSTEM_PROMPT, user_content)
        prompts.append(prompt)
        
    print(f"🚀 [vLLM 外部進程] 開始批次語意稽核 {len(prompts)} 篇論文...")
    temp_sampling_params = SamplingParams(temperature=0.0, max_tokens=256, stop=["<|im_end|>", "<|endoftext|>"])
    outputs = engine.generate(prompts, temp_sampling_params)
    
    scored_papers = []
    for paper, out in zip(papers, outputs):
        raw_text = out.outputs[0].text.strip()
        
        # 解析 LLM 回傳的 JSON 分數
        score = 0
        is_matching = False
        try:
            # 尋找 JSON 區塊並解析
            if "{" in raw_text and "}" in raw_text:
                json_str = raw_text[raw_text.find("{"):raw_text.rfind("}")+1]
                res = json.loads(json_str)
                score = int(res.get("related_score", 0))
                is_matching = bool(res.get("is_matching", False))
            else:
                res = json.loads(raw_text)
                score = int(res.get("related_score", 0))
                is_matching = bool(res.get("is_matching", False))
        except Exception:
            # 解析失敗時的保底機制：若有特定關鍵字則給低分
            if "redirect" in raw_text.lower() or "html" in raw_text.lower():
                score = 10
            else:
                score = 50 
        
        paper["llm_related_score"] = score
        paper["llm_is_matching"] = is_matching
        scored_papers.append(paper)
        
    # 根據分數從高到低排序，選出前 20 篇
    scored_papers.sort(key=lambda x: x.get("llm_related_score", 0), reverse=True)
    top_20_papers = scored_papers[:20]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(top_20_papers, f, ensure_ascii=False, indent=2)
    print(f"✅ [vLLM 外部進程] 成功選出分數最高的 20 篇論文並寫入 {output_path}")

if __name__ == "__main__":
    main()
