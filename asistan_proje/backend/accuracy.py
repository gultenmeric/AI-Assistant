from flask import Flask, request, jsonify, session
from sentence_transformers import SentenceTransformer
import faiss
import requests
import fitz
import pytesseract
from PIL import Image
import io
from flask_cors import CORS
import os
from datetime import timedelta


GEMINI_API_KEY = "AIz............................................."

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

def ask_gemini_with_context(user_query):
    best_match_indices = search_faiss(user_query)
    best_contexts = [global_chunks[i] for i in best_match_indices]
    prompt_text = (
        f"Öncelikle sen bir Çilek mobilya asistanısın. Kullanıcı şu soruyu sordu: '{user_query}'. "
        f"İşte en alakalı döküman bilgileri: {best_contexts}. Buna göre en doğru cevabı ver. "
        "Bu uygulamaya hakim asistan gibi davran. Kendin de bağlamı yakalamayı dene. "
        "Eğer soru yeterli değilse soruyu daha açık bir şekilde sormasını isteyebilirsin. "
        "PDF Belgesi dışında alakasız bir bilgiyi kesinlikle yanıtlama. "
        "Unutma,sen çilek mobilya asistanısın ve bu bilgilerle sınırlısın."
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    data = {"contents": [{"parts": [{"text": prompt_text}]}]}
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # HTTP hataları için kontrol
        response_json = response.json()
        if 'candidates' in response_json and response_json['candidates']:
            if 'content' in response_json['candidates'][0] and 'parts' in response_json['candidates'][0]['content']:
                parts = response_json['candidates'][0]['content']['parts']
                if parts and parts[0]['text']:
                    return parts[0]['text']
        return "Üzgünüm, bir sorun oluştu. Lütfen tekrar deneyin." #default değer
    except requests.exceptions.RequestException as e:
        print(f"İstek hatası: {e}")
        return "Üzgünüm, isteğiniz işlenirken bir hata oluştu."
    except Exception as e:
        print(f"Bilinmeyen hata: {e}")
        return "Üzgünüm, beklenmeyen bir hata oluştu."



def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "".join([page.get_text() for page in doc])
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
    for page in doc:
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            text_from_image = pytesseract.image_to_string(image, lang='tur')
            extracted_text += text_from_image + "\n"
    return extracted_text


pdf_path = "C:/Users/meric/OneDrive/Masaüstü/cilek\Çilek Portal App Kullanım Klavuzu.pdf"  
text = extract_text_from_pdf(pdf_path)
chunks = split_text_into_chunks(text)
global_chunks.extend(chunks)

image_text = extract_images_and_text_from_pdf(pdf_path)
image_chunks = split_text_into_chunks(image_text)
global_chunks.extend(image_chunks)

add_texts_to_index(global_chunks)

# API endpoint'i
app.secret_key = os.urandom(24)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

@app.route("/ask", methods=["POST"])
def ask():
    session.permanent = True
    if "chat_history" not in session:
        session["chat_history"] = []
    data = request.get_json()
    user_query = data["query"]
    chat_history = session["chat_history"]
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    full_prompt = f"Kullanıcının önceki mesajları:\n{history_text}\nSon soru: {user_query}"
    answer = ask_gemini_with_context(full_prompt)
    chat_history.append({"role": "user", "content": user_query})
    chat_history.append({"role": "assistant", "content": answer})
    session["chat_history"] = chat_history
    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True, port=5000) 
    

