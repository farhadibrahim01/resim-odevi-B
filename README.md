# Resim Ödevi B

Bu proje, yüz ifadelerine göre bireylerin ruhsal durumlarını sınıflandırmak amacıyla geliştirilmiştir. Eğitimde EfficientNet-B0 mimarisi kullanılarak dört sınıfa ayrılmış bir model eğitilmiştir:  
**bipolar**, **depression**, **healthy**, **schizophrenia**

## Klasör Yapısı

Resim-B/
├── data/
│ ├── raw/ → Tüm orijinal veriler
│ ├── train/ → Eğitim verileri
│ ├── val/ → Doğrulama verileri
│ └── test/ → Gerçek test verileri (model hiç görmedi)
├── outputs/ → Eğitilmiş modeller (örn: cnn_best.pt)
├── src/ → Eğitim, test ve veri hazırlama kodları
└── main.py → Ana çalıştırma dosyası

## Test Sonuçları

Accuracy : 100%
Precision : 0.99 - 1.00
Recall : 0.99 - 1.00
F1-Score : 0.99 - 1.00

**Confusion Matrix:**
[[1503 4 1 4] # bipolar
[ 3 909 0 0] # depression
[ 1 0 2278 0] # healthy
[ 0 0 1 682]] # schizophrenia


## Kullanılan Teknolojiler

- Python 3.11  
- PyTorch  
- torchvision  
- timm  
- scikit-learn  
- CUDA (GPU destekli eğitim)

  ## 🔧 Çalıştırma Talimatları

### 1. Ortam Kurulumu
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Veriyi Yerleştir
data/raw/ klasörüne 4 sınıfa ait klasörleri yerleştir:
bipolar/, depression/, healthy/, schizophrenia/


### 3. Veriyi Böl
```bash
python src/split_data.py
```

### 4. Boyutlandır
```bash
python src/resize_images.py
```

### 5. Modeli Eğit
```bash
python src/train_cnn.py
```

### 6. Modeli Test Et
```bash
python src/evaluate_test.py
```

Notlar:

Veri artırma (augmentation) şimdilik uygulanmamıştır. Bu aşama Resim C kapsamında analiz edilecektir.

Aşırı öğrenmeyi önlemek amacıyla yalnızca son katmanlar eğitilmiş ve erken durdurma kullanılmıştır.

Test verisi modelin hiç görmediği bağımsız görsellerden oluşur.

Farhad Ibrahimov
