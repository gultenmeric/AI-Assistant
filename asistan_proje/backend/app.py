from flask import Flask, request, jsonify, session
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from flask_cors import CORS  
from functools import lru_cache
import os
from requests.exceptions import RequestException
from datetime import timedelta# OpenAI ChatGPT entegrasyonu
from openai import OpenAI

client = OpenAI(api_key="sk-................................................................")  
app = Flask(__name__)
CORS(app)

model = SentenceTransformer('all-MiniLM-L6-v2')

index = faiss.IndexFlatL2(384)
global_chunks = []

def add_texts_to_index(texts):
    embeddings = model.encode(texts).astype('float32')
    index.add(embeddings)

def search_faiss(query):
    query_embedding = model.encode([query]).astype('float32').reshape(1, -1)
    D, I = index.search(query_embedding, k=6)
    return I[0]

def ask_openai_with_context(user_query, chat_history):
    best_match_indices = search_faiss(user_query)
    best_contexts = [global_chunks[i] for i in best_match_indices]
    context = "\n".join(best_contexts)

    # Benzerlik Skoru Hesaplama
    query_embedding = model.encode([user_query])
    context_embedding = model.encode(best_contexts)
    similarity_scores = np.dot(query_embedding, context_embedding.T).tolist()[0] #Cosine similarity yerine dot product kullanıldı.
    max_similarity = max(similarity_scores) if similarity_scores else 0 #max similarity değeri

    messages = [
        {"role": "system", "content": """Sen Çilek Mobilya için çalışan yardımsever bir asistansın.Aynı zamanda kullanıcının sorduğu her soruya cevap verebilecek bir asistansın. Çilek Mobilya ile ilgili olmasa da yanıt  vereceksin. 
    Görevin:
    1. Kullanıcının sorduğu genel sorulara, Normal sohbetlere ve selamlaşmalara doğal bir şekilde karşılık ver.
    2. Eğer kullanıcının sorusu özellikle Çilek Portal App, Çilekse, SSH veya sağlanan belge içerikleriyle ilgiliyse, sana verilen 'İlgili Belge İçeriği' bölümündeki bilgileri kullanarak cevap ver.
    3. **ÖNEMLİ:** Sağlanan 'İlgili Belge İçeriği' bölümünü YALNIZCA ve YALNIZCA kullanıcının sorusu bu belgelerdeki konularla DOĞRUDAN ilgiliyse kullan. Alakasız sorularda (selamlaşma, genel bilgi vb.) bu içeriği KESİNLİKLE dikkate alma ve cevaplarında bahsetme.
    4. Cevaplarını her zaman doğal, samimi ve nazik bir dilde ver."""}
    ]
    messages.extend(chat_history)

    # Eğer benzerlik düşükse, bağlamı gönderme
    if max_similarity < 0.1: #benzerlik eşiği
        messages.append({"role": "user", "content": f"Kullanıcı Sorusu: {user_query}"})
    else:
        messages.append({
            "role": "user",
            "content": f"Kullanıcı Sorusu: {user_query}\n\nİlgili Belge İçeriği:\n{context}"
        })

    try:
        response = client.chat.completions.create(
           model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API hatası: {e}")
        return "Bir hata oluştu, lütfen tekrar deneyin."

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

def extract_images_and_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    extracted_text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            text_from_image = pytesseract.image_to_string(image, lang='tur')
            extracted_text += text_from_image + "\n"
    return extracted_text

pdf_path ="C:/Users/meric/OneDrive/Masaüstü/cilek/Çilek Portal App Kullanım Klavuzu.pdf"

text = extract_text_from_pdf(pdf_path)
chunks = split_text_into_chunks(text)
global_chunks.extend(chunks)

image_text = extract_images_and_text_from_pdf(pdf_path)
image_chunks = split_text_into_chunks(image_text)
global_chunks.extend(image_chunks)

add_texts_to_index(global_chunks)

# 9. API endpoint
app.secret_key = os.urandom(24)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

@app.route("/ask2", methods=["POST"])
def ask():
    session.permanent = True

    if "chat_history" not in session:
        session["chat_history"] = []

    data = request.get_json()
    user_query = data["query"]
    chat_history = session["chat_history"]

    answer = ask_openai_with_context(user_query, chat_history)

    chat_history.append({"role": "user", "content": user_query})
    chat_history.append({"role": "assistant", "content": answer})
    session["chat_history"] = chat_history

    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
