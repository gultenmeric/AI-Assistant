# Python imaj 
FROM python:3.10

# Çalışma dizini 
WORKDIR /app

# Sadece requirements.txt 
COPY requirements.txt .

# Bağımlılıklar
RUN pip install --no-cache-dir --timeout 10000 -r requirements.txt

# Sadece match.py 
COPY match.py .

#  çalıştır
CMD ["python", "match.py"]