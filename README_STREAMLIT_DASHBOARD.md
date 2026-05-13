# Streamlit Dashboard Rehberi

Bu dokuman, proje icindeki Streamlit arayuzunun ne yaptigini, nasil calistirilacagini, hangi veri dosyalarini kullandigini ve sayfa bazli olarak nasil yorumlanacagini detayli sekilde aciklar.

## 1) Amac

Bu dashboard, kredi risk tahmin projesinin sonucunu teknik ve teknik olmayan paydaslar icin ayni ekranda anlatmak icin tasarlanmistir.

Ana hedefler:
- EDA bulgularini interaktif gorsellerle gostermek
- ML model karsilastirmasini sade bir sekilde sunmak
- Is maliyeti odakli threshold kararini aciklamak
- Sunum sirasinda hizli ve okunabilir bir deneyim saglamak

## 2) Teknoloji ve Mimari

Kullanilan stack:
- Python
- Streamlit
- Pandas
- NumPy
- Plotly

Arayuz dosyasi:
- app.py (tek dosya calisma mantigi)

Veri akis ozeti:
- output/step7/dashboard_pack altindaki CSV dosyalari birincil kaynaktir
- EDA histogram ve ek analizler icin data/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv dosyasi kullanilir
- output/step7/dashboard_pack yoksa output/step6 fallback olarak denenir

## 3) Gereksinimler

- Python 3.11+
- requirements.txt icindeki bagimliliklar

Kurulum:

```bash
pip install -r requirements.txt
```

## 4) Calistirma

```bash
streamlit run app.py
```

Acilan adres:
- http://localhost:8501

## 5) Zorunlu Veri Dosyalari

Arayuz asagidaki dosyalari bekler (dashboard pack):

- model_metrics_long.csv
- model_comparison.csv
- feature_importance_gbt.csv
- feature_importance_rf.csv
- confusion_matrix_best.csv
- roc_curve_best.csv
- threshold_tuning_best_model.csv
- business_cost_matrix.csv
- kpi_cards.csv
- loan_status_distribution.csv
- monthly_default_trend.csv
- fico_default_risk.csv
- home_ownership_risk.csv
- loan_purpose_risk.csv

Eksik dosya varsa uygulama acik hata mesaji verir ve devam etmez.

## 6) Arayuz Kontrolleri (Sidebar)

Sidebar uzerindeki kontroller:
- Dark Mode: koyu/acik tema gecisi
- Page: sayfa secimi
- Model Filter: secili modeller uzerinden filtreleme
- Date Filter: tarih araligi secimi (monthly trend icin)
- Threshold Slider: threshold odakli analizler icin secim

## 7) Sayfa Bazli Icerik

### 7.1 Executive Overview

Amac:
- Ust duzey KPI ve portfoy ozetini gostermek

Icerik:
- KPI kartlari (Total Loans, Default Rate, Best Model, Best AUC, Avg Interest Rate)
- Loan Status Distribution (pie)
- Monthly Default Trend (line)
- Risk Profile Insight karti
- Onerilen threshold karti (Cost ve F1)

Ne anlatir:
- Portfoyun genel risk profilini
- Temerrutun zaman davranisini
- Is/teknik kararlar arasindaki dengeyi

### 7.2 EDA Dashboard

Amac:
- Veri dagilimlari, segment riskleri ve iliski sinyallerini gostermek

Icerik:
- Histogramlar: loan_amnt, annual_inc, dti
- Loan Status Pie
- Monthly Default Trend
- FICO Risk Analysis
- Home Ownership Risk
- Loan Purpose Risk
- Missing Value Summary
- Correlation Heatmap
- Ek EDA gorselleri:
  - DTI by Default Status
  - Income vs Loan Amount
  - Interest Rate by Default

Yorumlama standardi:
- Her grafikte iki satir vardir:
  - Yorum
  - Eksen aciklamasi (X, Y ve varsa renk/Z anlami)

### 7.3 Machine Learning Dashboard

Amac:
- Model performansini ve aciklanabilirligi sunmak

Icerik:
- Model Metrics Comparison (grouped bar)
- Feature Importance (Horizontal Bar Chart)
  - Model secimi: GBT veya RF
  - Top-N slider: gosterilecek ozellik adedi
- ROC Curve
- Confusion Matrix Heatmap
- Model Detail Table
- Best Model Highlight karti

Not:
- Feature importance dataseti mevcut olarak GBT ve RF icin bulunur
- Diger modellerde (LR, LinearSVC, DecisionTree) importance kaynagi yoksa chartta gosterilmez

### 7.4 Threshold and Business Strategy Dashboard

Amac:
- Is maliyeti odakli threshold kararini netlestirmek

Icerik:
- Recommended Threshold kartlari
- Threshold vs F1
- Threshold vs Expected Cost
- FP/FN Cost Impact
- Business Cost Matrix Heatmap

Ne anlatir:
- Skor optimizasyonu ile is maliyeti optimizasyonunun her zaman ayni olmadigini
- FN maliyetinin baskin oldugu durumda daha farkli bir threshold secimi gerekebilecegini

## 8) Tema ve Tasarim

Temel tasarim prensipleri:
- Modern dark/light hybrid tema
- Glassmorphism KPI kartlari
- Yalnizca gerekli renk vurgulari
- Yuksek okunabilirlik (sunum odakli)
- Responsive kolon yerlesimi

Renk ailesi:
- Arka plan: #0B1220
- Kart: #13263A
- Metin: #EAF2FF
- Vurgu mavi: #2D8CFF
- Risk kirmizi: #E85D5D
- Basari yesil: #2FBF71
- Uyari turuncu: #F2A93B

## 9) Performans ve Dayaniklilik

Uygulama tarafinda dayaniklilik ozellikleri:
- CSV yuklemeleri cache edilir (st.cache_data)
- Eksik dosya kontrolu vardir
- EDA ham veri yoksa ilgili bloklar kontrollu sekilde bilgi mesaji verir
- Yukleme sirasinda spinner gosterilir

## 10) MLflow ile Iliski

Bu arayuz, MLflow run sonuclarini dogrudan mlruns klasorunden okumaz.
Arayuz, step6/step7 export edilen CSV dosyalarina dayanir.

MLflow gereksinimi proje akisinda su sekilde saglanir:
- Tum deneylerin parametre/metric/model bilgileri step6 tarafinda MLflow'a loglanir
- Dashboard bu ciktilarin raporlanmis ozetini gosterir

Eger canli MLflow run listesi ve artifact tarama istenirse:
- Arayuze ek bir Model Governance sayfasi eklenerek mlruns klasoru veya MLflow tracking URI dogrudan okunabilir

## 11) Sik Karsilasilan Sorunlar

1. Uygulama acilmiyor
- Bagimliliklari tekrar kur:
  - pip install -r requirements.txt
- Sonra tekrar calistir:
  - streamlit run app.py

2. Missing required files hatasi
- output/step7/dashboard_pack altinda tum zorunlu CSV dosyalarinin oldugunu kontrol et

3. EDA bazi grafikler bos
- Ham veri dosyasinin varligini kontrol et:
  - data/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv

4. Sayfa ustu kesik gorunuyor
- Tarayicida hard refresh yap (Ctrl+F5)

## 12) Gelistirme Notlari

Kod organizasyonu:
- app.py tek dosyada calisacak sekilde duzenlenmistir
- Sayfa bazli render fonksiyonlari ayriktir
- Ortak stil ve yardimci fonksiyonlar tekrar kullanilir

Gelistirme yaparken onerilen adimlar:
1. Yeni veri dosyasini REQUIRED_FILES listesine ekle
2. prepare_data icinde tip donusumlerini tanimla
3. Yeni sayfa/fonksiyon ekle
4. Sidebar page listesine yeni sayfayi ekle
5. main() icindeki page router'a yeni sayfayi bagla

## 13) Sunum Ipuclari

- Executive sayfasi ile basla, genel resmi ver
- EDA ile riskin kimde yogunlastigini anlat
- ML sayfasinda model secim gerekcesini metriklerle destekle
- Threshold sayfasinda teknik optimum ile is optimumu farkini netlestir
- Kapanista operasyonel karar ciktisini tek cümlede ver:
  - onerilen threshold, beklenen etki, izleme plani

---

Bu README sadece Streamlit arayuzunu kapsar.
Projenin tum pipeline detaylari icin ana dokumanlara bak:
- README.md
- README_STEP4_STEP5.md
- README_STEP6_STEP7.md
