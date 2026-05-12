# Step 4 ve Step 5 Uygulama Rehberi

Bu dokuman, proje kapsamindaki Step 4 (EDA) ve Step 5 (Feature Engineering) adimlarini bastan sona dogru sekilde calistirmak, beklenen ciktilari almak ve teslim kriterlerini karsilamak icin hazirlanmistir.

## 1) Kapsam ve Teslim Kriterleri

### Step 4 (EDA)
- Veri dagilimi, null analizi, hedef dagilimi, korelasyon ve segment bazli incelemeler yapilmis olmali.
- Is birimleri icin anlamli en az birkac bulgu metinle aciklanmali.
- Notebook hatasiz calismali ve gorseller uretilmeli.

### Step 5 (Feature Engineering)
- En az 5 feature uretilmeli (notebookta 8 adet var).
- Feature secim mantigi aciklanmali (notebookta markdown bolumu var).
- Gold feature tablosu Delta Lake formatinda yazilmali.

## 2) On Kosullar

- Docker Desktop acik olmali.
- Python ortami hazir olmali.
- Veri dosyasi su yolda bulunmali:
  - data/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv

Opsiyonel ama onerilen:
- Docker servislerini kaldir:

```bash
make start
```

## 3) Kullanilan Notebooklar

- Step 4: notebooks/step4_eda.ipynb
- Step 5: notebooks/step5_feature_engineering.ipynb

## 4) Step 4 Calistirma Akisi

1. notebooks/step4_eda.ipynb dosyasini ac.
2. Kernel secimini dogrula.
3. Tum hucreleri sirayla calistir.
4. Su kontrolleri yap:
   - Veri basariyla okunuyor mu?
   - Hedef degisken dagilimi geliyor mu?
   - Temel EDA grafiklerinin tamami olusuyor mu?
   - Notebook sonunda ozet yorumlar gorunuyor mu?

### Step 4 Definition of Done
- Tum kod hucreleri hatasiz.
- En az temel EDA grafikleri uretilmis.
- Is acisindan yorumlar mevcut.

## 5) Step 5 Calistirma Akisi (Strict Delta Mod)

Bu projede Step 5 assignment kriteri icin strict Delta mod aktif ayarlanmistir.

### Varsayilan Davranis
- STEP5_STRICT_DELTA_REQUIRED=1 (varsayilan)
- Delta runtime acilamazsa notebook hata verir.
- Gold cikti Delta formatinda yazilmazsa adim tamamlanmis sayilmaz.

### Calistirma Sirasi
1. notebooks/step5_feature_engineering.ipynb dosyasini ac.
2. Tum hucreleri yukaridan asagiya sirayla calistir.
3. Su satirlari outputta kontrol et:
   - Delta runtime available: True
   - Strict Delta required: True
4. Gold write asamasinda su kontrolleri yap:
   - Gold feature table format: delta
   - Gold feature table path: .../delta_lake/gold/features

## 6) Step 5 Uretilen Featurelar

Notebookta uretilen ana engineered featurelar:
- income_to_loan_ratio
- credit_stress_score
- fico_bucket
- high_risk_purpose_flag
- employment_stability_score
- credit_utilization_risk
- installment_income_ratio
- delinquency_risk_score

Not: Teslim kriterindeki minimum 5 feature kosulu saglanmistir.

## 7) Ortam Degiskenleri (Step 5)

Asagidaki degiskenler gerektiğinde notebook davranisini ayarlar:

- STEP5_STRICT_DELTA_REQUIRED
  - 1: Delta zorunlu (onerilen ve assignment modu)
  - 0: Delta zorunlu degil
- STEP5_MAX_NOTEBOOK_ROWS
  - Local calismada ornekleme limiti (varsayilan: 450000)
- STEP5_SAMPLE_SEED
  - Ornekleme seed degeri (varsayilan: 42)
- STEP5_ENABLE_FULL_COUNT
  - 1 ise pipeline sonrasi full count alir (agir olabilir)

## 8) Siklikla Karsilasilan Sorunlar ve Cozum

### Delta runtime acilmiyor
- Belirti: Delta-enabled Spark session could not start
- Cozum:
  - Docker/Spark Linux ortaminda calistir.
  - Delta ve Spark surum uyumunu kontrol et.
  - Java surumunu kontrol et (genellikle 11 veya 17 tercih edilir).

### Py4J baglanti hatalari (WinError 10054/10061)
- Neden: JVM crash, bellek baskisi veya lokal ortam siniri.
- Cozum:
  - Ornekleme acik kalsin (varsayilan zaten acik).
  - Full count kapali kalsin.
  - Driver memory ve shuffle ayarlarini asiri yukseltme.

### RandomForest maxBins hatasi
- Neden: Kategorik index kardinalitesi maxBins degerini asar.
- Durum: Notebookta dinamik maxBins hesabi ile cozuldu.

## 9) Teslim Oncesi Son Kontrol Listesi

### Step 4 Checklist
- [ ] Tum hucreler hatasiz tamamlandi
- [ ] EDA grafikleri ve ozet yorumlar mevcut

### Step 5 Checklist
- [ ] Tum hucreler hatasiz tamamlandi
- [ ] En az 5 feature uretilmis (8 feature var)
- [ ] Feature mantigi markdown ile aciklanmis
- [ ] Gold tablo Delta formatinda yazilmis
- [ ] Gold yolunda veri okunabiliyor

## 10) Hizli Dogrulama Kodlari

Step 5 sonunda Delta tablosunu okuyup satir sayisini kontrol etmek icin:

```python
check_df = spark.read.format("delta").load(f"{DELTA_BASE}/gold/features")
print(check_df.count())
check_df.printSchema()
```

Bu kod hata vermez ve satir getirirse Step 5 Delta teslim kriteri teknik olarak saglanmis demektir.
