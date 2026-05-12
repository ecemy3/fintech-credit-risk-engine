# Step 6 ve Step 7 Uygulama Rehberi

Bu dokuman, Step 6 (ML pipeline) ve Step 7 (dashboard ve export) tarafinda yapilan tum adimlari, gelistirmeleri, runtime sorununu ve Docker tabanli stabil cozum yolunu kapsar.

## 1) Kapsam ve Hedef

Step 6 hedefi:
- Gold feature tablosundan 5 farkli classification modeli egitmek
- Modelleri Accuracy, F1, Precision, Recall, AUC ile karsilastirmak
- MLflow ile parametre, metrik ve model artifact takibini yapmak

Step 7 hedefi:
- Sonuclari dashboard formatinda gorsellestirmek
- Power BI icin temiz ve sayfa bazli veri paketini export etmek

## 2) Yapilan Tum Gelistirmeler

### Step 6 notebook (notebooks/step6_ml_models.ipynb)
- Delta Gold table okuma akisi eklendi.
- Windows local runtime sorunlari icin Delta/parquet fallback stratejisi eklendi.
- Spark ayarlarinda Arrow kapatildi (spark.sql.execution.arrow.pyspark.enabled=false) ve toPandas kaynakli platform hatalari azaltildi.
- VectorAssembler ile numeric featurelardan features vektoru olusturuldu.
- Train/Test split kuruldu (0.8 / 0.2, seed=42).
- 5 model egitildi:
  - LogisticRegression
  - DecisionTreeClassifier
  - RandomForestClassifier
  - GBTClassifier
  - LinearSVC
- Metrikler hesaplandi:
  - Accuracy
  - F1
  - Precision
  - Recall
  - AUC ROC
- MLflow entegrasyonu aktif:
  - model parametreleri loglandi
  - metrikler loglandi
  - model artifactleri loglandi
- Feature importance ciktilari eklendi (RF ve GBT).
- Confusion matrix ve ROC curve eklendi.
- Threshold tuning eklendi (0.05-0.95 arasi).
- Business cost matrix eklendi (farkli FP/FN maliyet senaryolari).
- Step 7 icin export dosyalari eklendi.

### Step 7 notebook (notebooks/step7_dashboard_export.ipynb)
- Step 6 artifactlerini okuyacak veri girisi kuruldu.
- Delta/parquet fallback ile feature table okuma dayanikli hale getirildi.
- 4 sayfa dashboard uretimi:
  - page1_executive_overview.png
  - page2_eda_dashboard.png
  - page3_ml_dashboard.png
  - page4_threshold_strategy.png
- Power BI icin optimize export paketi olusturuldu.
- Long format model metrik tablosu eklendi (model_metrics_long.csv).
- KPI kart datasi eklendi (kpi_cards.csv).
- Gorsel-veri esleme manifesti eklendi (visual_manifest.csv).

## 3) Runtime Sorunu ve Kok Neden

Windows local Spark notebook kernelinde asagidaki tipte hata goruldu:
- NativeIO / winutils / access0 tabanli RuntimeError

Bu hata, yerel Windows Spark-Hadoop native katman uyumsuzlugu nedeniyle olusur.

Kontrol sonucunda:
- delta_lake/gold/features mevcut
- delta_lake/gold/features_parquet_fallback mevcut
- third_party/hadoop/bin icinde winutils.exe mevcut

Yani veri mevcut olmasina ragmen local native katman hatasi olusabiliyor.

## 4) Stabil Cozum (Docker Spark Container)

En stabil yontem:
1. docker compose up -d
2. Spark container icinde notebook akisini calistir

Linux container ortaminda:
- NativeIO hatasi yok
- winutils bagimliligi yok
- Delta okuma/yazma daha stabil

Yapilan dogrulama:
- Spark container icinde Delta okuma basarili
- okunan satir sayisi: 271765
- kolon sayisi: 27
- label dagilimi dogru geldi

## 5) Gercek Calisma Sonuclari (Step 6)

Kaynak dosya: output/step6/model_comparison.csv

Model performans ozet:
- GBTClassifier: auc=0.7125, f1=0.4438, accuracy=0.6530
- RandomForestClassifier: auc=0.7096, f1=0.4409, accuracy=0.6555
- LogisticRegression: auc=0.7037, f1=0.4372, accuracy=0.6576
- LinearSVC: auc=0.7012, f1=0.4299, accuracy=0.6796
- DecisionTreeClassifier: auc=0.5886, f1=0.4348, accuracy=0.6383

Best model (AUC):
- GBTClassifier

Kaynak dosya: output/step6/executive_metrics.json
- total_loans: 271765
- default_rate: 0.21195
- best_model: GBTClassifier
- best_auc: 0.71252
- avg_interest_rate: 13.2867

Threshold ve maliyet ozet:
- best threshold by cost (FP=1, FN=5): 0.4
- best threshold by f1: 0.5

Kaynak dosyalar:
- output/step6/threshold_tuning_best_model.csv
- output/step6/business_cost_matrix.csv
- output/step6/threshold_summary.json

## 6) Uretilen Dosyalar

### Step 6 output
- output/step6/model_comparison.csv
- output/step6/confusion_matrix_best.csv
- output/step6/roc_curve_best.csv
- output/step6/feature_importance_rf.csv
- output/step6/feature_importance_gbt.csv
- output/step6/threshold_tuning_best_model.csv
- output/step6/business_cost_matrix.csv
- output/step6/threshold_summary.json
- output/step6/executive_metrics.json

### Step 7 output
- output/step7/page1_executive_overview.png
- output/step7/page2_eda_dashboard.png
- output/step7/page3_ml_dashboard.png
- output/step7/page4_threshold_strategy.png

### Step 7 dashboard pack
- output/step7/dashboard_pack/model_comparison.csv
- output/step7/dashboard_pack/model_metrics_long.csv
- output/step7/dashboard_pack/confusion_matrix_best.csv
- output/step7/dashboard_pack/roc_curve_best.csv
- output/step7/dashboard_pack/feature_importance_rf.csv
- output/step7/dashboard_pack/feature_importance_gbt.csv
- output/step7/dashboard_pack/threshold_tuning_best_model.csv
- output/step7/dashboard_pack/business_cost_matrix.csv
- output/step7/dashboard_pack/kpi_cards.csv
- output/step7/dashboard_pack/loan_status_distribution.csv
- output/step7/dashboard_pack/monthly_default_trend.csv
- output/step7/dashboard_pack/fico_default_risk.csv
- output/step7/dashboard_pack/home_ownership_risk.csv
- output/step7/dashboard_pack/loan_purpose_risk.csv
- output/step7/dashboard_pack/visual_manifest.csv
- output/step7/dashboard_pack/executive_metrics.json

## 7) Power BI Sayfa Esleme Onerisi

1. Executive Overview
- kpi_cards.csv
- loan_status_distribution.csv

2. EDA Dashboard
- monthly_default_trend.csv
- fico_default_risk.csv
- home_ownership_risk.csv
- loan_purpose_risk.csv

3. ML Dashboard
- model_metrics_long.csv
- feature_importance_rf.csv veya feature_importance_gbt.csv
- confusion_matrix_best.csv
- roc_curve_best.csv

4. Threshold Strategy
- threshold_tuning_best_model.csv
- business_cost_matrix.csv

## 8) Sonuc

Step 6 ve Step 7 tarafi proje seviyesinde tamamlandi:
- Spark MLlib tabanli 5 model karsilastirmasi
- MLflow takip
- threshold tuning + business cost matrix
- dashboard gorselleri
- Power BI icin paket export
- Docker Spark runtime ile stabil calisma modeli
