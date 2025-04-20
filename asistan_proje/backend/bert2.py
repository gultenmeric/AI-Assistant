from flask import Flask, request, jsonify, session
from sentence_transformers import SentenceTransformer
import faiss
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np
import requests
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from flask_cors import CORS
import os
from datetime import timedelta
from openai import OpenAI
from pathlib import Path
#Dosya dinamik,ancak gpt cevapları geliştirillmeli
# Uygulama ve CORS ayarları
app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

# API Anahtarları
GEMINI_API_KEY = "AIz..................."
OPENAI_API_KEY = "sk-................................."

# PDF Klasör Yolu


PDF_FOLDER = os.path.join(os.path.dirname(__file__), 'pdfs')  # Proje içinde pdfs klasörü


os.makedirs(PDF_FOLDER, exist_ok=True)

class PDFManager:
    def __init__(self):
        self.pdf_files = []
        self.load_pdfs()
        self.setup_watcher()
    
    def load_pdfs(self):
        """PDF klasöründeki tüm dosyaları yükle"""
        try:
            self.pdf_files = [
                os.path.join(PDF_FOLDER, f) 
                for f in os.listdir(PDF_FOLDER) 
                if f.lower().endswith('.pdf')
            ]
            print(f"Yüklenen PDF'ler: {self.pdf_files}")
        except NotADirectoryError:
            print(f"HATA: {PDF_FOLDER} geçerli bir klasör değil!")
            self.pdf_files = []

# Modeller ve veritabanları
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
gemini_index = faiss.IndexFlatL2(768)
openai_index = faiss.IndexFlatL2(768)
gemini_chunks = []
openai_chunks = []

# İstemciler
client = OpenAI(api_key=OPENAI_API_KEY)

# PDF Yöneticisi Sınıfı
class PDFManager:
    def __init__(self):
        self.pdf_files = []
        self.load_pdfs()
        self.setup_watcher()
    
    def load_pdfs(self):
        """PDF klasöründeki tüm dosyaları yükle"""
        self.pdf_files = [
            os.path.join(PDF_FOLDER, f) 
            for f in os.listdir(PDF_FOLDER) 
            if f.lower().endswith('.pdf')
        ]
        print(f"Yüklenen PDF'ler: {self.pdf_files}")
    
    def setup_watcher(self):
        """Dosya sistemindeki değişiklikleri izle"""
        event_handler = PDFHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, PDF_FOLDER, recursive=False)
        self.observer.start()
        print("PDF izleyici başlatıldı")

class PDFHandler(FileSystemEventHandler):
    def __init__(self, pdf_manager):
        self.pdf_manager = pdf_manager
    
    def on_created(self, event):
        if event.src_path.lower().endswith('.pdf'):
            self.pdf_manager.load_pdfs()  # Yeniden yükle
            print(f"Yeni PDF algılandı: {event.src_path}")
pdf_manager = PDFManager()

# Ortak Fonksiyonlar
def add_texts_to_index(texts, index):
    embeddings = model.encode(texts).astype('float32')
    index.add(embeddings)

def search_faiss(query, index):
    query_embedding = model.encode([query]).astype('float32').reshape(1, -1)
    D, I = index.search(query_embedding, k=6)
    return I[0]

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
            #print(text)
        return text
    except Exception as e:
        print(f"PDF okuma hatası ({pdf_path}): {e}")
        return ""

def extract_images_and_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        extracted_text = ""
        for page in doc:
            images = page.get_images(full=True)
            for img in images:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image = Image.open(io.BytesIO(base_image["image"]))
                extracted_text += pytesseract.image_to_string(image, lang='tur') + "\n"
                #print(extracted_text)
        return extracted_text
    except Exception as e:
        print(f"PDF görsel işleme hatası ({pdf_path}): {e}")
        return ""

def split_text_into_chunks(text, chunk_size=256):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def process_pdfs(pdf_paths):
    all_chunks = []
    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text_into_chunks(text)
        all_chunks.extend(chunks)
        
        image_text = extract_images_and_text_from_pdf(pdf_path)
        if image_text.strip():
            all_chunks.extend(split_text_into_chunks(image_text))
    return all_chunks

def refresh_indices():
    """Tüm PDF'leri yeniden işle ve indeksle"""
    global gemini_chunks, openai_chunks
    
    # Önceki verileri temizle
    gemini_chunks = []
    openai_chunks = []
    gemini_index.reset()
    openai_index.reset()
    
    # Yeni PDF'leri işle
    chunks = process_pdfs(pdf_manager.pdf_files)
    gemini_chunks.extend(chunks)
    openai_chunks.extend(chunks)
    add_texts_to_index(chunks, gemini_index)
    add_texts_to_index(chunks, openai_index)
    
    print(f"Indices refreshed. Total chunks: {len(chunks)}")
    #print(openai_chunks)
refresh_indices()


def ask_gemini(user_query):
    best_match_indices = search_faiss(user_query, gemini_index)
    best_contexts = [gemini_chunks[i] for i in best_match_indices]

    prompt_text = f"""
Sen Çilek Mobilya çalışanları için geliştirilmiş yardımsever ve bilgili bir asistansın.  
Ancak aynı zamanda genel bilgiye de sahipsin ve kullanıcıların her türlü sorusuna yanıt verebilirsin.  

**Kullanıcı Sorusu:** {user_query}  

Aşağıda soruyla ilgili bağlam bilgisi verilmiştir.  
Eğer bu bilgiler soruyla ilgiliyse kullan, eğer ilgisizse kendi genel bilgi havuzundan yanıt ver.  

**Bağlam Bilgisi:**  
{best_contexts}  

Unutma:  
- Eğer Çilek Mobilya, Çilek Portal App, Çilekse veya SSH ile ilgili herhangi bir soru gelirse, öncelikle bağlam bilgisini kullan. Eşleşiyorsa direkt bağlamdan cevap ver. 
- Eğer soru Çilek Mobilya ile ilgili değilse, genel bilgi havuzunu kullanarak en iyi yanıtı ver.  
- Eğer emin değilsen, kullanıcıdan daha fazla detay iste.  
-Soruyu soran kişiye herhangi bir bağlamdan beslendiğini belli etme.Mesela "Bağlamda bu yok" gibi.Bunları söyleme.
Şimdi, kullanıcıya net ve doğru bir yanıt ver.  
"""
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    response = requests.post(url, json={"contents": [{"parts": [{"text": prompt_text}]}]})
    
    try:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except:
        return "Üzgünüm, bir hata oluştu."

def ask_openai(user_query, chat_history):
    best_match_indices = search_faiss(user_query, openai_index)
    best_contexts = [openai_chunks[i] for i in best_match_indices]
    context = "\n".join(best_contexts)

    messages = [
        {"role": "system", "content": """Sen Çilek Mobilya çalışanları için üretilmiş yardımsever
         ve kibar bir asistansın. 
         Kullanıcının sorduğu her soruya cevap verebilecek bir asistansın.
         Eğer bilmediğin veya emin olamadığın bir soru ise soran kişiden daha detaylı bir soru isteyebilirsin. 
         Çilek Portal App, Çilekse ve SSH hakkında bilgi sahibisin."""},
        *chat_history,
        {"role": "user", "content": f"Soru: {user_query}\n\nSoruyu yanıtlamak için Bağlam bilgisi:\n{context}"}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.8,
            #presence_penalty=0.3,  #yanıtta geçen kelimeleri tekrar kullanmamasını sağlar ama riskli bence
            top_p=0.7 #yuksekse bağlama bağımlılık azalır temp e alternatif olarak kullanılır.
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI hatası: {e}")
        return "Bir hata oluştu."


@app.route("/ask/gemini", methods=["POST"])
def gemini_endpoint():
    session.permanent = True
    if "gemini_history" not in session:
        session["gemini_history"] = []

    data = request.get_json()
    user_query = data["query"]
    answer = ask_gemini(user_query)  #!!!!!!!!!!!!!

    session["gemini_history"].append({"role": "user", "content": user_query})
    session["gemini_history"].append({"role": "assistant", "content": answer})
    return jsonify({"response": answer})

@app.route("/ask/openai", methods=["POST"])
def openai_endpoint():
    session.permanent = True
    if "openai_history" not in session:
        session["openai_history"] = []

    data = request.get_json()
    user_query = data["query"]
    answer = ask_openai(user_query, session["openai_history"])

    session["openai_history"].append({"role": "user", "content": user_query})
    session["openai_history"].append({"role": "assistant", "content": answer})
    return jsonify({"response": answer})

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    """Yeni PDF yükleme endpoint'i"""
    if 'file' not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Dosya seçilmedi"}), 400
    
    if file and file.filename.lower().endswith('.pdf'):
        file_path = os.path.join(PDF_FOLDER, file.filename)
        file.save(file_path)
        
        # Watchdog otomatik olarak yenileme yapacak
        return jsonify({
            "status": "success",
            "message": "PDF başarıyla yüklendi. Sistem otomatik olarak güncellenecek.",
            "file_path": file_path
        })
    
    return jsonify({"error": "Geçersiz dosya formatı"}), 400

@app.route("/refresh", methods=["POST"])
def manual_refresh():
    """Manuel yenileme endpoint'i"""
    refresh_indices()
    return jsonify({
        "status": "success",
        "message": "Veritabanı başarıyla güncellendi",
        "total_pdfs": len(pdf_manager.pdf_files),
        "total_chunks": len(gemini_chunks)
    })

if __name__ == "__main__":
    try:
        app.run(debug=True, port=5030)
    finally:
        pdf_manager.observer.stop()
        pdf_manager.observer.join()
        
      