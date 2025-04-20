import os
import sys
import io
import logging
from logging.handlers import RotatingFileHandler
from datetime import timedelta
import numpy as np
import requests
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import faiss

from flask import Flask, request, jsonify, session
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from openai import OpenAI

# --- Uygulama ve CORS Ayarları ---
app = Flask(__name__)
CORS(app) # Tüm kaynaklardan gelen isteklere izin ver 
app.secret_key = os.urandom(24) # Güvenli anahtar 
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30) # Oturum süresi

GEMINI_API_KEY = "AI..........."
OPENAI_API_KEY = "sk-..............."

# --- PDF Klasör Yolu ---
PDF_FOLDER = os.path.join(os.path.dirname(__file__), 'pdfs')

# --- Logging Kurulumu ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s')
log_file = 'rag_application.log'

# Dosya Handler (Dönen)
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8') # 10MB limit, 5 backup
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO) # Dosyaya INFO ve üzeri logları yaz

# Konsol Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.DEBUG) # Konsola DEBUG ve üzerini yaz (daha detaylı)

# Ana Logger
logger = logging.getLogger('RAGAppLogger')
logger.setLevel(logging.DEBUG) # İşlenecek en düşük seviye

# Handler'ları ekle (Flask debug modunda çift eklemeyi önle)
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler) # Konsol çıktısı

logger.info("="*50)
logger.info("Uygulama başlatılıyor...")
logger.info(f"PDF Klasörü: {PDF_FOLDER}")
logger.info("Loglama sistemi başarıyla başlatıldı.")
logger.info("="*50)

# --- Modeller ve Veritabanları ---
try:
    logger.info("Embedding modeli yükleniyor: 'paraphrase-multilingual-mpnet-base-v2'")
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    embedding_dim = model.get_sentence_embedding_dimension()
    logger.info(f"Embedding modeli başarıyla yüklendi. Boyut: {embedding_dim}")
except Exception as e:
    logger.critical(f"Embedding modeli yüklenirken KRİTİK HATA: {e}", exc_info=True)
    # Model yüklenemezse uygulama başlamamalı
    raise RuntimeError(f"Embedding modeli yüklenemedi: {e}")

# FAISS Index'leri Embedding boyutuna göre oluşturdum
gemini_index = faiss.IndexFlatIP(embedding_dim)
openai_index = faiss.IndexFlatIP(embedding_dim)
gemini_chunks = []
openai_chunks = []
logger.info(f"FAISS Index'leri {embedding_dim} boyutlu olarak başlatıldı.")

# --- OpenAI İstemcisi
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI istemcisi başarıyla başlatıldı.")
except Exception as e:
    logger.error(f"OpenAI istemcisi başlatılırken hata: {e}")

# --- PDF İşleme Fonksiyonları ---

def extract_text_from_pdf(pdf_path):
    """PDF dosyasından metin içeriğini çıkarır."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num, page in enumerate(doc):
            text += page.get_text("text") # "text" argümanı daha temiz çıktı verebilir
        doc.close()
        logger.debug(f"'{os.path.basename(pdf_path)}' metin çıkarma başarılı. Uzunluk: {len(text)}")
        return text
    except Exception as e:
        logger.error(f"PDF metin okuma hatası ({os.path.basename(pdf_path)}): {e}", exc_info=False)
        return ""

def extract_images_and_text_from_pdf(pdf_path):
    """PDF dosyasındaki görsellerden OCR ile metin çıkarır."""
    extracted_text = ""
    try:
        doc = fitz.open(pdf_path)
        img_count = 0
        for page_num, page in enumerate(doc):
            images = page.get_images(full=True)
            img_count += len(images)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    # Belki burada basit bir ön işleme (gri tonlama vb.) faydalıydı belki ama değerleri tutturamadım.
                    ocr_text = pytesseract.image_to_string(image, lang='tur') 
                    if ocr_text.strip():
                        extracted_text += ocr_text.strip() + "\n\n" 
                        logger.debug(f"  - Sayfa {page_num+1}, Resim {img_index+1} OCR başarılı.")
                except pytesseract.TesseractNotFoundError:
                     logger.error("Tesseract kurulu değil veya PATH'de bulunamıyor! OCR işlemi yapılamıyor.")
                     # Bu hatayı bir kere loglamak yeterli olabilir.
                     # raise # Ya da hatayı yükseltip işlemi durdur
                     doc.close()
                     return "" # Tesseract yoksa devam etme
                except Exception as img_e:
                    logger.warning(f"  - Sayfa {page_num+1}, Resim {img_index+1} işlenirken/OCR sırasında hata: {img_e}", exc_info=False)
        doc.close()
        if img_count > 0:
             logger.debug(f"'{os.path.basename(pdf_path)}' görsel metin çıkarma tamamlandı. Toplam {img_count} resim bulundu. Çıkarılan metin uzunluğu: {len(extracted_text)}")
        return extracted_text
    except Exception as e:
        logger.error(f"PDF görsel işleme hatası ({os.path.basename(pdf_path)}): {e}", exc_info=False)
        return ""

def split_text_into_chunks(text, chunk_size=256, chunk_overlap=50):
    """Metni belirlenen boyut ve çakışma ile parçalara ayırır."""
    if not text or not text.strip():
        return []

    words = text.split() # boşluklara göre ayır
    if len(words) == 0:
        return []

    chunks = []
    if len(words) <= chunk_size:
        single_chunk = " ".join(words).strip()
        return [single_chunk] if single_chunk else []

    step_size = chunk_size - chunk_overlap
    if step_size <= 0:
        step_size = max(1, chunk_size // 2)
        logger.warning(f"Chunk boyutu ({chunk_size}) ile overlap ({chunk_overlap}) uyumsuz. Step size {step_size} olarak ayarlandı.")

    for i in range(0, len(words), step_size):
        chunk_words = words[i:min(i + chunk_size, len(words))] # Sona gelince taşmayı önle
        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if i + chunk_size >= len(words): # Eğer son parçayı kapsadıysak çık
            break

    # Overlap nedeniyle oluşabilecek tam kopyaları temizle (basit yöntem)
    final_chunks = []
    seen_chunks = set()
    for chunk in chunks:
        if chunk not in seen_chunks:
            final_chunks.append(chunk)
            seen_chunks.add(chunk)

    logger.debug(f"Metin {len(words)} kelimeden {len(final_chunks)} chunk'a bölündü (size={chunk_size}, overlap={chunk_overlap})")
    return final_chunks

def process_pdfs(pdf_paths):
    """Verilen PDF yollarındaki metinleri ve görsellerden OCR ile metinleri çıkarıp chunk'lar."""
    logger.info(f"Toplam {len(pdf_paths)} PDF işlenmeye başlanıyor...")
    all_chunks = []
    chunk_size = 120  # Ana chunk boyutu
    chunk_overlap = 50 # Çakışma miktarı

    for pdf_path in pdf_paths:
        base_filename = os.path.basename(pdf_path)
        logger.info(f"İşleniyor: '{base_filename}'")
        text_chunks_count = 0
        img_chunks_count = 0

        # Metin Çıkarma ve Chunking
        text = extract_text_from_pdf(pdf_path)
        if text.strip():
            chunks = split_text_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            text_chunks_count = len(chunks)
            all_chunks.extend(chunks)
            logger.info(f"  - Metin: {text_chunks_count} chunk çıkarıldı.")
        else:
            logger.info(f"  - Metin: İçerik bulunamadı veya çıkarılamadı.")

        # Görsel Metni (OCR) Çıkarma ve Chunking
        image_text = extract_images_and_text_from_pdf(pdf_path)
        if image_text.strip():
            img_chunks = split_text_into_chunks(image_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            img_chunks_count = len(img_chunks)
            all_chunks.extend(img_chunks)
            logger.info(f"  - Görsel Metni (OCR): {img_chunks_count} chunk çıkarıldı.")
        else:
             logger.info(f"  - Görsel Metni (OCR): İçerik bulunamadı veya çıkarılamadı.")

        logger.info(f"-> '{base_filename}' tamamlandı. Toplam chunk: {text_chunks_count + img_chunks_count}")

    logger.info(f"PDF işleme tamamlandı. Toplam chunk (tekilleştirme öncesi): {len(all_chunks)}")
    # Basit tekilleştirme (aynı metne sahip chunkları kaldır)
    unique_chunks = list(dict.fromkeys(all_chunks))
    logger.info(f"PDF işleme tamamlandı. Tekil chunk sayısı: {len(unique_chunks)}")
    return unique_chunks


# --- İndeksleme ve Arama ---

def add_texts_to_index(texts, index):
    """Verilen metin listesini encode edip FAISS index'ine ekler."""
    if not texts:
        logger.warning("FAISS'e eklenecek metin bulunamadı.")
        return
    try:
        logger.info(f"{len(texts)} adet metin embedding'i hesaplanıyor...")
        embeddings = model.encode(texts, show_progress_bar=True).astype('float32') # İlerleme çubuğu ekle
        logger.info("Embedding hesaplama tamamlandı. FAISS index'ine ekleniyor...")
        index.add(embeddings)
        logger.info(f"{len(texts)} embedding başarıyla index'e eklendi. Index toplam boyutu: {index.ntotal}")
    except Exception as e:
        logger.error(f"Metinler index'e eklenirken hata: {e}", exc_info=True)

def search_faiss(query, index, k=3):
    """FAISS index'inde benzerlik araması yapar, indeksleri ve mesafeleri döndürür."""
    if index.ntotal == 0:
        logger.warning("FAISS index boş, arama yapılamıyor.")
        return [], [] # Boş listeler döndür

    logger.debug(f"FAISS araması: Sorgu='{query[:100]}...', k={k}")
    try:
        query_embedding = model.encode([query]).astype('float32').reshape(1, -1)
        # D: Mesafeler (L2 - düşük = daha benzer), I: İndeksler
        distances, indices = index.search(query_embedding, k=min(k, index.ntotal)) # k, index boyutundan büyük olamaz
        logger.debug(f"FAISS sonuçları: İndeksler={indices[0]}, Mesafeler={distances[0]}")
        return indices[0], distances[0]
    except Exception as e:
        logger.error(f"FAISS araması sırasında hata: {e}", exc_info=True)
        return [], [] # Hata durumunda boş listeler döndür

def refresh_indices():
    """Tüm PDF'leri yeniden işler ve FAISS indekslerini günceller."""
    logger.info("="*30 + " İndeks Yenileme Başlatıldı " + "="*30)
    global gemini_chunks, openai_chunks, gemini_index, openai_index

    # Önceki verileri temizle
    gemini_chunks = []
    openai_chunks = []
    # Index'i sıfırlamak yerine yeniden oluşturmak daha temiz olabilir
    embedding_dim = model.get_sentence_embedding_dimension()
    gemini_index = faiss.IndexFlatL2(embedding_dim)
    openai_index = faiss.IndexFlatL2(embedding_dim)
    logger.info("Mevcut chunk listeleri ve FAISS indeksleri temizlendi/sıfırlandı.")

    # Yeni PDF'leri işle (process_pdfs zaten loglama yapıyor)
    chunks = process_pdfs(pdf_manager.pdf_files)

    if chunks:
        gemini_chunks.extend(chunks)
        openai_chunks.extend(chunks)
        logger.info(f"{len(chunks)} tekil chunk belleğe yüklendi.")

        # İndekslere ekle (add_texts_to_index zaten loglama yapıyor)
        logger.info("Gemini index'ine ekleniyor...")
        add_texts_to_index(chunks, gemini_index)
        logger.info("OpenAI index'ine ekleniyor...")
        add_texts_to_index(chunks, openai_index)
    else:
         logger.warning("İşlenecek PDF bulunamadı veya PDF'lerden chunk çıkarılamadı. İndeksler boş bırakıldı.")

    logger.info(f"İndeks yenileme tamamlandı. Toplam tekil chunk: {len(gemini_chunks)}")
    logger.info(f"Gemini Index Boyutu: {gemini_index.ntotal}, OpenAI Index Boyutu: {openai_index.ntotal}")
    logger.info("="*30 + " İndeks Yenileme Tamamlandı " + "="*30)


# --- Asistan Fonksiyonları (Detaylı Loglama ile) ---

def ask_gemini(user_query):
    """Gemini kullanarak RAG işlemi yapar ve cevabı döndürür."""
    logger.info(f"--- [Gemini Süreci Başladı] ---")
    logger.info(f"Kullanıcı Sorusu: '{user_query}'")

    try:
        # 1. Benzerlik Arama ve Chunk'ları Alma
        k_results = 3 # Kaç chunk getireceğimizi belirle
        best_match_indices, distances = search_faiss(user_query, gemini_index, k=k_results)

        if not best_match_indices.size: # Eğer arama sonucu boşsa (index boş veya hata)
             logger.warning(f"[Gemini] '{user_query}' için FAISS'te chunk bulunamadı (index boş veya arama hatası).")
             return "Üzgünüm, bu soruyla ilgili bilgi veritabanımızda bulunamadı."

        # Alınan chunk'ların metinlerini ve mesafelerini al (indeks geçerliliğini kontrol etmeye gerek yok, FAISS sadece var olanı döndürür)
        retrieved_chunks_text = [gemini_chunks[i] for i in best_match_indices]
        distances_for_log = distances # Zaten sadece bulunanların mesafesi

        logger.info(f"Alınan Chunk İndeksleri (FAISS): {best_match_indices}")
        logger.info(f"İlgili Chunk Mesafeleri (Düşük = İyi): {distances_for_log}")

        # Loglamak için context'i formatla
        context_for_log = "\n\n---\n\n".join([
            f"Chunk Index {idx} (Mesafe: {dist:.4f}):\n{text[:500]}..." # Log şişmesini önlemek için kısalt
            for idx, dist, text in zip(best_match_indices, distances_for_log, retrieved_chunks_text)
        ])
        logger.info(f"LLM'e Gönderilecek Alınan Bağlam (Context) Önizlemesi:\n{context_for_log}")

        # LLM'e gönderilecek tam context
        context_for_llm = "\n\n".join(retrieved_chunks_text)

        # 2. Prompt Hazırlama
        prompt_text = f"""
Sen Çilek Mobilya çalışanları için geliştirilmiş yardımsever ve bilgili bir asistansın.
Ancak aynı zamanda genel bilgiye de sahipsin ve kullanıcıların her türlü sorusuna yanıt verebilirsin.

**Kullanıcı Sorusu:** {user_query}

Aşağıda soruyla ilgili olması muhtemel bağlam bilgisi (Chunk'lar) verilmiştir.
Eğer bu bilgiler soruyla doğrudan ilgiliyse, cevabını oluştururken ÖNCELİKLE bu bilgileri kullan.
Eğer bilgiler ilgisiz görünüyorsa veya soruyu tam yanıtlamıyorsa, kendi genel bilgi havuzundan yanıt ver.

**Bağlam Bilgisi:**
{context_for_llm}

Unutma:
- Çilek Mobilya, Çilek Portal App, Çilekse veya SSH ile ilgili sorularda bağlamı önceliklendir.
- Cevabında "bağlamda belirtildiği gibi" veya "chunk'lara göre" gibi ifadeler KULLANMA. Doğrudan cevap ver.
- Emin değilsen, daha fazla detay iste.

Şimdi, kullanıcıya net, doğru ve kibar bir yanıt ver.
"""
        logger.debug(f"[Gemini] LLM'e gönderilecek TAM PROMPT (kısaltılmış olabilir):\n{prompt_text[:1000]}...") # DEBUG için

        # 3. LLM API Çağrısı
        logger.info("[Gemini] API isteği gönderiliyor (gemini-1.5-flash)...")
        # Gemini 1.5 Flash modelini kullanalım
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
        headers = {'Content-Type': 'application/json'}

        response = requests.post(url, headers=headers, json=payload, timeout=60) # Timeout ekle
        response.raise_for_status() # HTTP hatalarını yakala (4xx, 5xx)

        # 4. Cevabı Alma ve Loglama
        raw_response_json = response.json()
        logger.debug(f"[Gemini] Ham API Yanıtı:\n{raw_response_json}") # DEBUG için

        try:
            answer = raw_response_json['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError, TypeError) as e:
             logger.error(f"[Gemini] API yanıt formatı beklenenden farklı! Hata: {e}. Yanıt: {raw_response_json}")
             return "Üzgünüm, Gemini'den gelen yanıt işlenirken bir sorun oluştu."

        logger.info(f"[Gemini] Üretilen Cevap:\n{answer}")
        logger.info(f"--- [Gemini Süreci Tamamlandı] ---")
        return answer

    except requests.exceptions.Timeout:
        logger.error("[Gemini] API isteği zaman aşımına uğradı.")
        logger.info(f"--- [Gemini Süreci Hata İle Sonlandı] ---")
        return "Üzgünüm, cevap alınırken bir zaman aşımı sorunu yaşandı."
    except requests.exceptions.RequestException as e:
        logger.error(f"[Gemini] API isteği sırasında hata: {e}", exc_info=True)
        # Hata yanıtını logla (varsa)
        try: logger.error(f"[Gemini] Hata anındaki API yanıtı (varsa): {e.response.text}")
        except: pass
        logger.info(f"--- [Gemini Süreci Hata İle Sonlandı] ---")
        return "Üzgünüm, Gemini API ile iletişim kurarken bir bağlantı hatası oluştu."
    except Exception as e:
        logger.error(f"[Gemini] Süreç sırasında beklenmeyen bir hata oluştu: {e}", exc_info=True)
        logger.info(f"--- [Gemini Süreci Hata İle Sonlandı] ---")
        return "Üzgünüm, isteğiniz işlenirken beklenmedik bir hata oluştu."

def ask_openai(user_query, chat_history):
    """OpenAI kullanarak RAG işlemi yapar ve cevabı döndürür."""
    logger.info(f"--- [OpenAI Süreci Başladı] ---")
    logger.info(f"Kullanıcı Sorusu: '{user_query}'")
    logger.info(f"Mevcut Sohbet Geçmişi Uzunluğu: {len(chat_history)}")

    try:
        # 1. Benzerlik Arama ve Chunk'ları Alma
        k_results = 3
        best_match_indices, distances = search_faiss(user_query, openai_index, k=k_results)

        if not best_match_indices.size:
             logger.warning(f"[OpenAI] '{user_query}' için FAISS'te chunk bulunamadı.")
             retrieved_chunks_text = []
             context_for_prompt = "İlgili bağlam bulunamadı."
             logger.info("LLM'e gönderilecek bağlam bulunamadı.")
        else:
            retrieved_chunks_text = [openai_chunks[i] for i in best_match_indices]
            distances_for_log = distances

            logger.info(f"Alınan Chunk İndeksleri (FAISS): {best_match_indices}")
            logger.info(f"İlgili Chunk Mesafeleri (Düşük = İyi): {distances_for_log}")

            context_for_log = "\n\n---\n\n".join([
                f"Chunk Index {idx} (Mesafe: {dist:.4f}):\n{text[:500]}..." # Log için kısalt
                for idx, dist, text in zip(best_match_indices, distances_for_log, retrieved_chunks_text)
            ])
            logger.info(f"LLM'e Gönderilecek Alınan Bağlam (Context) Önizlemesi:\n{context_for_log}")
            context_for_prompt = "\n\n".join(retrieved_chunks_text) # LLM için tam context

        # 2. Prompt (Mesaj Listesi) Hazırlama
        system_message = {
            "role": "system",
            "content": """Sen Çilek Mobilya çalışanları için üretilmiş yardımsever ve kibar bir asistansın. Kullanıcının sorduğu her soruya cevap verebilecek bir asistansın.
            Sana sağlanan 'Bağlam Bilgisi'ni kullanarak soruyu yanıtlamaya çalış. Eğer bağlam soruyu yanıtlamak için yeterli veya ilgili değilse, kendi genel bilgini kullan.
            Cevaplarında 'bağlama göre', 'chunk'ta yazdığı gibi' ifadeler kullanma. Doğrudan cevap ver.
            Eğer bilmediğin veya emin olamadığın bir soru ise soran kişiden daha detaylı bilgi isteyebilirsin.
            Çilek Portal App, Çilekse ve SSH hakkında bilgi sahibisin ve bu konularda öncelikle sağlanan bağlamı kullanmalısın."""
        }

        user_message_content = f"Kullanıcı Sorusu: {user_query}\n\nSoruyu yanıtlamak için kullanabileceğin Bağlam Bilgisi:\n{context_for_prompt}"

        messages = [system_message] + chat_history + [{"role": "user", "content": user_message_content}]

        logger.debug(f"[OpenAI] LLM'e gönderilecek TAM MESAJ LİSTESİ:\n{messages}") # DEBUG için

        # 3. LLM API Çağrısı
        logger.info("[OpenAI] API isteği gönderiliyor (gpt-3.5-turbo)...") # Veya gpt-4-turbo
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Model seçimi
            messages=messages,
            temperature=0.5, # Daha tutarlı cevaplar için
            # max_tokens=1000 # İsteğe bağlı: Cevap uzunluğunu sınırlamak için
            timeout=60.0 # Timeout ekle
        )

        # 4. Cevabı Alma ve Loglama
        logger.debug(f"[OpenAI] Ham API Yanıtı:\n{response}") # DEBUG için

        answer = response.choices[0].message.content

        # Token kullanımını loglamak faydalı
        if response.usage:
            logger.info(f"[OpenAI] Token kullanımı: Prompt={response.usage.prompt_tokens}, Completion={response.usage.completion_tokens}, Total={response.usage.total_tokens}")

        logger.info(f"[OpenAI] Üretilen Cevap:\n{answer}")
        logger.info(f"--- [OpenAI Süreci Tamamlandı] ---")
        return answer

    except Exception as e: # Daha spesifik OpenAI hatalarını yakalamak daha iyi (örn: openai.APIError, openai.Timeout)
        logger.error(f"[OpenAI] Süreç sırasında hata oluştu: {e}", exc_info=True)
        # Hatanın tipine göre daha spesifik mesajlar verilebilir
        logger.info(f"--- [OpenAI Süreci Hata İle Sonlandı] ---")
        return "Üzgünüm, OpenAI ile iletişim kurarken bir hata oluştu."


# --- PDF Yönetimi (Watchdog ile) ---

class PDFManager:
    """PDF dosyalarını yönetir ve değişiklikleri izler."""
    def __init__(self):
        self.pdf_files = []
        self.load_pdfs() # Başlangıçta yükle
        self.setup_watcher()

    def load_pdfs(self):
        """PDF klasöründeki tüm dosyaları listeler."""
        try:
            self.pdf_files = [
                os.path.join(PDF_FOLDER, f)
                for f in os.listdir(PDF_FOLDER)
                if os.path.isfile(os.path.join(PDF_FOLDER, f)) and f.lower().endswith('.pdf')
            ]
            logger.info(f"Yüklü PDF listesi güncellendi: {len(self.pdf_files)} dosya bulundu.")
            logger.debug(f"Yüklenen PDF Dosyaları: {[os.path.basename(p) for p in self.pdf_files]}")
        except NotADirectoryError:
            logger.error(f"HATA: Belirtilen PDF yolu bir klasör değil: {PDF_FOLDER}")
            self.pdf_files = []
        except Exception as e:
            logger.error(f"PDF listesi yüklenirken hata: {e}", exc_info=True)
            self.pdf_files = []

    def setup_watcher(self):
        """Dosya sistemi değişikliklerini izlemek için Watchdog'u ayarlar."""
        event_handler = PDFHandler(self)
        self.observer = Observer()
        try:
            self.observer.schedule(event_handler, PDF_FOLDER, recursive=False) # Sadece ana klasörü izle
            self.observer.start()
            logger.info(f"'{PDF_FOLDER}' klasörü için PDF izleyici başarıyla başlatıldı.")
        except Exception as e:
             logger.error(f"PDF izleyici başlatılırken hata: {e}. Klasör izlenemeyecek.", exc_info=True)


class PDFHandler(FileSystemEventHandler):
    """PDF klasöründeki dosya olaylarını işler."""
    def __init__(self, pdf_manager):
        self.pdf_manager = pdf_manager

    def process(self, event):
        """Ortak olay işleme mantığı."""
        # Sadece .pdf dosyalarıyla ilgilen
        if not event.is_directory and event.src_path.lower().endswith('.pdf'):
            filename = os.path.basename(event.src_path)
            logger.info(f"PDF Klasöründe Değişiklik Algılandı: Olay={event.event_type}, Dosya='{filename}'")
            # PDF listesini güncelle ve indeksleri yenile
            self.pdf_manager.load_pdfs()
            # Otomatik yenileme: Yeni PDF eklendiğinde veya değiştirildiğinde tetikle
            refresh_indices()

    def on_created(self, event):
        self.process(event)

    def on_modified(self, event):
         # Değiştirilme olayları bazen birden fazla tetiklenebilir, dikkatli olunmalı
         # Şimdilik değiştirilmeyi de yenileme sebebi olarak alalım
        self.process(event)

    def on_deleted(self, event):
        # Silinen dosyayı listeden çıkarıp yenileme
        if not event.is_directory and event.src_path.lower().endswith('.pdf'):
            filename = os.path.basename(event.src_path)
            logger.info(f"PDF Silindi: '{filename}'. İndeksler yenileniyor...")
            self.pdf_manager.load_pdfs()
            refresh_indices()

# --- PDF Yöneticisini Başlat ---
pdf_manager = PDFManager()


# --- API Endpoint'leri ---

@app.route("/ask/gemini", methods=["POST"])
def gemini_endpoint():
    session.permanent = True # Oturumun süresi dolmadan kalıcı olmasını sağla
    # Gemini için sohbet geçmişi (isteğe bağlı, şu anki ask_gemini desteklemiyor)
    # if "gemini_history" not in session:
    #     session["gemini_history"] = []
    #     logger.debug("[Gemini Endpoint] Yeni oturum için geçmiş başlatıldı.")

    data = request.get_json()
    if not data or "query" not in data:
         logger.warning("[Gemini Endpoint] Geçersiz istek: JSON veya 'query' alanı eksik.")
         return jsonify({"error": "Geçersiz istek formatı. 'query' alanı içeren JSON gönderin."}), 400

    user_query = data["query"]
    if not user_query.strip():
        logger.warning("[Gemini Endpoint] Boş sorgu alındı.")
        return jsonify({"error": "Sorgu boş olamaz"}), 400

    # ask_gemini zaten detaylı loglama yapıyor
    answer = ask_gemini(user_query)

    # Oturum geçmişi yönetimi (şimdilik kapalı)
    # session["gemini_history"].append({"role": "user", "content": user_query})
    # session["gemini_history"].append({"role": "assistant", "content": answer})
    # session.modified = True
    # logger.debug(f"[Gemini Endpoint] Oturum geçmişi güncellendi.")

    return jsonify({"response": answer})

@app.route("/ask/openai", methods=["POST"])
def openai_endpoint():
    session.permanent = True
    if "openai_history" not in session:
        session["openai_history"] = []
        logger.debug("[OpenAI Endpoint] Yeni oturum için geçmiş başlatıldı.")

    data = request.get_json()
    if not data or "query" not in data:
         logger.warning("[OpenAI Endpoint] Geçersiz istek: JSON veya 'query' alanı eksik.")
         return jsonify({"error": "Geçersiz istek formatı. 'query' alanı içeren JSON gönderin."}), 400

    user_query = data["query"]
    if not user_query.strip():
        logger.warning("[OpenAI Endpoint] Boş sorgu alındı.")
        return jsonify({"error": "Sorgu boş olamaz"}), 400

    # ask_openai zaten detaylı loglama yapıyor
    current_history = session.get("openai_history", []) # Geçmişi al
    answer = ask_openai(user_query, current_history) # Geçmişi fonksiyona gönder

    # Geçmişi güncelle
    current_history.append({"role": "user", "content": user_query})
    current_history.append({"role": "assistant", "content": answer})
    # Geçmiş çok uzarsa eski mesajları silmek gerekebilir
    MAX_HISTORY = 20 # Örneğin son 10 konuşmayı tut (10 user + 10 assistant)
    if len(current_history) > MAX_HISTORY:
        logger.debug(f"OpenAI geçmişi {MAX_HISTORY} mesajı aştı, eski mesajlar siliniyor.")
        current_history = current_history[-MAX_HISTORY:]

    session["openai_history"] = current_history
    session.modified = True # Oturumun güncellendiğini belirt
    logger.debug(f"[OpenAI Endpoint] Oturum geçmişi güncellendi. Toplam mesaj: {len(current_history)}")

    return jsonify({"response": answer})

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    """Yeni PDF yükleme endpoint'i."""
    if 'file' not in request.files:
        logger.warning("[Upload PDF] İstekte 'file' kısmı bulunamadı.")
        return jsonify({"error": "Dosya bulunamadı"}), 400

    file = request.files['file']
    if file.filename == '':
        logger.warning("[Upload PDF] Dosya seçilmedi (filename boş).")
        return jsonify({"error": "Dosya seçilmedi"}), 400

    if file and file.filename.lower().endswith('.pdf'):
        # Güvenlik: Dosya adını temizle (örn: ../ gibi yolları engelle)
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        file_path = os.path.join(PDF_FOLDER, filename)

        try:
            file.save(file_path)
            logger.info(f"PDF başarıyla yüklendi ve kaydedildi: {file_path}")
            # Watchdog otomatik olarak algılayıp refresh_indices() çağıracak
            return jsonify({
                "status": "success",
                "message": f"'{filename}' başarıyla yüklendi. İndeksler otomatik olarak güncellenecek.",
                "filename": filename
            }), 201 # 201 Created
        except Exception as e:
            logger.error(f"PDF kaydedilirken hata oluştu: {file_path} - Hata: {e}", exc_info=True)
            return jsonify({"error": "Dosya kaydedilirken sunucu hatası oluştu"}), 500
    else:
        logger.warning(f"[Upload PDF] Geçersiz dosya formatı veya dosya yok: {file.filename}")
        return jsonify({"error": "Geçersiz dosya formatı. Sadece PDF dosyaları kabul edilir."}), 400

@app.route("/refresh", methods=["POST"])
def manual_refresh():
    """Manuel olarak indeksleri yenileme endpoint'i."""
    logger.info("Manuel indeks yenileme isteği alındı.")
    try:
        refresh_indices()
        return jsonify({
            "status": "success",
            "message": "Veritabanı ve indeksler başarıyla güncellendi.",
            "total_pdfs": len(pdf_manager.pdf_files),
            "total_chunks_in_memory": len(gemini_chunks), # veya openai_chunks
            "gemini_index_size": gemini_index.ntotal,
            "openai_index_size": openai_index.ntotal
        })
    except Exception as e:
        logger.error(f"Manuel yenileme sırasında hata: {e}", exc_info=True)
        return jsonify({"error": "İndeksler yenilenirken bir hata oluştu."}), 500

# --- Uygulamayı Başlatma ---
if __name__ == "__main__":
    # Başlangıçta indeksleri bir kere yenileyelim
    logger.info("Uygulama başlangıcında indeksler yenileniyor...")
    refresh_indices()
    logger.info("Başlangıç indeks yenilemesi tamamlandı.")

    try:
        # host='0.0.0.0' 
        logger.info("Flask sunucusu başlatılıyor (port=5030, debug=True)...")
        app.run(host='0.0.0.0', port=5030, debug=True)
    except Exception as e:
         logger.critical(f"Flask sunucusu başlatılamadı: {e}", exc_info=True)
    finally:
        # Uygulama kapanırken observer'ı temizle
        if pdf_manager and pdf_manager.observer.is_alive():
            logger.info("Flask sunucusu durduruluyor. PDF izleyici kapatılıyor...")
            pdf_manager.observer.stop()
            pdf_manager.observer.join()
            logger.info("PDF izleyici başarıyla kapatıldı.")
        logger.info("Uygulama kapatıldı.")