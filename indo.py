# pip install fasttext
# 下載官方壓縮版模型 (僅 9MB，速度快到不可思議)
# wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
# nltk.download('punkt')
# nltk.download('punkt_tab')
import nltk
from nltk.tokenize import sent_tokenize
import fasttext
import os


model_path = "lid.176.ftz"
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

# 測試那篇地雷論文
title = "Classification of Melanoma Skin Cancer using Convolutional Neural Network."
abstract = "Melanoma cancer is a type of skin cancer and is the most dangerous one because it causes the most of skin cancer deaths. Melanoma comes from melanocyte cells, melanin-producing cells, so that melanomas are generally brown or black coloured. Melanomas are mostly caused by exposure to ultraviolet radiation that damages the DNA of skin cells. The diagnoses of melanoma cancer are often performed manually by using visuals of the skilled doctors, analyzing the result of dermoscopy examination and match it with medical sciences. Manual detection weakness is highly influenced by human subjectivity that makes it inconsistent in certain conditions. Therefore, a computer assisted technology is needed to help classifying the results of dermoscopy examination and to deduce the results more accurately with a relatively faster time. The making of this application starts with problem analysis, design, implementation, and testing. This application uses deep learning technology with Convolutional Neural Network method and LeNet-5 architecture for classifying image data. The experiment using 44 images data from the training results with a different number of training and epoch resulted the highest percentage of success at 93% in training and 100% in testing, which the number of training data used of 176 images and 100 epochs. This application was created using Python programming language and Keras library as Tensorflow back-end."
is_title_english= has_non_english(title, ft_model, threshold=0.3)
is_abstract_english = has_non_english(abstract, ft_model, threshold=0.5)
if is_title_english*is_abstract_english==False:  
    lang_reason = "FastText detected non-English content."
else:
    lang_reason = "FastText passed English detection."
print(f"Title English: {is_title_english} | Abstract English: {is_abstract_english} | Reason: {lang_reason}")
# #Klasifikasi Citra Tanaman Perdu Liar Berkhasiat Obat Menggunakan Jaringan Syaraf Tiruan Radial Basis Function

