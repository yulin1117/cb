# pip install fasttext
# 下載官方壓縮版模型 (僅 9MB，速度快到不可思議)
# wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
# nltk.download('punkt')
# nltk.download('punkt_tab')
import nltk
from nltk.tokenize import sent_tokenize
import fasttext
import os


model_path = "/home/yuzh952h/workspaces/horse/yuzh952h-cb/CitationBias/lid.176.ftz"
ft_model = fasttext.load_model(model_path)

def has_non_english(text, model, threshold=0.5):
    sentences = sent_tokenize(text)

    for s in sentences:
        s = s.strip()
        print(f"🔍 Checking sentence: '{s}'") 
        if len(s) < 5:
            continue

        labels, probs = model.predict(s, k=1)
        print(f"   Predicted: {labels[0]} | Confidence: {probs[0]:.4f}")
        lang = labels[0].replace("__label__", "")
        conf = probs[0]
        # ❗ 只要任何句子最高預測不是英文就false
        if lang != "en" :
            print(f"⚠️ Detected non-English sentence: '{s}' | Predicted: {lang} | Confidence: {conf:.4f}")
            return False
        # ❗ 只要任何句子是英文但信心度過低就false
        if lang == "en" and conf < threshold:
            print(f"⚠️ Detected low-confidence English sentence: '{s}' | Predicted: {lang} | Confidence: {conf:.4f}")
            return False

    return True
def title_lang_detect(text: str, model,threshold: float = 0.5) -> bool:
    """
    Sentence-by-sentence validation using FastText.
    Returns True if non-English sentence ratio < threshold, else False.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    if len(text) < 5:
        return False
    (labels, probs) = model.predict(text.strip().replace("\n", " "), k=1)
    
    lang = labels[0].replace("__label__", "")
    confidence = probs[0]  # 這裡拿到的才是真正的 float 機率值 (e.g. 0.85)

    if lang != "en":
        return False
        
    if confidence < threshold:
        return False

    return True
def abstract_lang_detect(text, model, threshold=0.2):
    """
    Sentence-by-sentence validation using FastText. Filters out documents containing more non-eng sentences than threshold.
    """
    if not text or not text.strip():
        return False
    sentences = sent_tokenize(text)
    valid_sentences_count=0
    non_eng_sentences  = 0  
    for s in sentences:
        s = s.strip().replace("\n", " ")
        if len(s) < 5:
            continue
        valid_sentences_count += 1
        labels, probs = model.predict(s, k=1)
        lang = labels[0].replace("__label__", "")

        if lang != "en" :
            non_eng_sentences += 1
    if valid_sentences_count > 0:
        if non_eng_sentences / valid_sentences_count >= threshold:
                return False        
    return True
# 測試那篇地雷論文
title = "Dieta com baixo teor de carboidrato: um estudo de caso"
abstract = "Introduction: The best composition of the diet in the treatment of obesity is still quite controversial, considering that the ideal composition for weight loss and maintenance is still unknown, and among this search for the best strategy, it has been highlighted as the most popular currently. carbohydrate-restricted diets. More and more, new types of diets and nutritional strategies have been sought to solve these problems caused by obesity and each professional is looking for a nutrition line to base himself on. Materials and Methods: A case study with a qualitative approach, with the nature of applied research, carried out for a month with an individual who was obese and submitted to a diet with 25% of carbohydrates per day. Results: There was a loss of six kg, a 3.5% reduction in fat percentage, four cm in the waist, five cm in the abdominal and four cm in the hips. All biochemical parameters had a considerable improvement, but there was a reduction in total cholesterol, which reduced 26.33%, followed by blood glucose, which obtained a reduction of 21.73%. Triglycerides had a reduction of 21.13%. Conclusion: It becomes more evident that a low-carbohydrate diet is an effective alternative for weight loss, lipid and glycemic profile improvement."
is_title_english= title_lang_detect(title, ft_model, threshold=0.5)
is_abstract_english = abstract_lang_detect(abstract, ft_model, threshold=0.2)
if is_title_english*is_abstract_english==False:  
    lang_reason = "FastText detected non-English content."
else:
    lang_reason = "FastText passed English detection."
print(f"Title English: {is_title_english} | Abstract English: {is_abstract_english} | Reason: {lang_reason}")
# #Klasifikasi Citra Tanaman Perdu Liar Berkhasiat Obat Menggunakan Jaringan Syaraf Tiruan Radial Basis Function

