.PHONY: install build up down restart streaming logs clean help

# Yerel geliştirme için tüm kütüphaneleri kur
install:
	pip install -r requirements.txt

# Docker image'larını build et
build:
	docker-compose build

# Tüm servisleri başlat
up:
	docker-compose up -d

# Tüm servisleri başlat (logları terminalde göster)
up-logs:
	docker-compose up

# Tüm servisleri durdur
down:
	docker-compose down

# Tüm servisleri yeniden başlat
restart:
	docker-compose down && docker-compose up -d

# Spark Streaming job'ı çalıştır (up'tan sonra)
streaming:
	docker exec spark python /app/spark/streaming_job.py

# Jupyter Notebook sunucusunu baslat (http://localhost:8888)
jupyter:
	docker exec spark pip install jupyter notebook -q
	docker exec -d spark jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' --notebook-dir=/app/notebooks
	@echo "Jupyter hazir: http://localhost:8888"

# Tüm container loglarını göster
logs:
	docker-compose logs -f

# Belirli servisin loglarını göster (örn: make logs-producer)
logs-producer:
	docker-compose logs -f producer

logs-kafka:
	docker-compose logs -f kafka

logs-spark:
	docker-compose logs -f spark-master

# Delta Lake ve checkpoint'leri temizle (dikkatli kullan!)
clean-data:
	rm -rf delta_lake/ mlruns/ mlartifacts/

# Tüm Docker artifact'larını temizle
clean-docker:
	docker-compose down -v --rmi local

# Build + up birlikte
start: build up
	@echo "Servisler baslatildi."
	@echo "Kafka UI  : http://localhost:8080"
	@echo "Spark UI  : http://localhost:8081"
	@echo "MLflow UI : http://localhost:5000"

help:
	@echo "Kullanilabilir komutlar:"
	@echo "  make install       - Python kutuphanelerini kur (yerel)"
	@echo "  make build         - Docker image'larini build et"
	@echo "  make up            - Servisleri arka planda baslat"
	@echo "  make up-logs       - Servisleri baslat (log gorunumu ile)"
	@echo "  make down          - Servisleri durdur"
	@echo "  make restart       - Servisleri yeniden baslat"
	@echo "  make streaming     - Spark Streaming job'i calistir"
	@echo "  make logs          - Tum loglari goster"
	@echo "  make logs-producer - Producer loglarini goster"
	@echo "  make start         - build + up + link listesi"
	@echo "  make clean-data    - Delta Lake ve MLflow verilerini sil"
	@echo "  make clean-docker  - Docker container/image'larini temizle"
