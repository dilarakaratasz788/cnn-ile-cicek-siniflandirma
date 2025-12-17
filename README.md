# 🌸 CNN ile Çiçek Sınıflandırma (Flower Classification)

Bu proje, TensorFlow ve Keras kütüphanelerini kullanarak çiçek resimlerini yüksek doğrulukla sınıflandırmak için bir **Konvolüsyonel Sinir Ağı (CNN)** modeli eğitir. 

## 🎯 Projenin Amacı
`tf_flowers` veri setini kullanarak; papatya, karahindiba, gül, ayçiçeği ve lale türlerini birbirinden ayırt edebilen derin öğrenme tabanlı bir sistem geliştirmek.

## 🚀 Öne Çıkan Teknik Özellikler
* **Veri Artırma (Data Augmentation):** Modelin farklı açılardan ve ışık koşullarından gelen resimleri tanıması için rastgele çevirme, parlaklık ve kontrast ayarları uygulandı.
* **Optimizasyon:** Eğitim sürecini hızlandırmak ve tıkanmaları önlemek için `ReduceLROnPlateau` ve `EarlyStopping` mekanizmaları kullanıldı.
* **Performans:** `tf.data` API'si ve `prefetch` kullanılarak veri yükleme hatları optimize edildi.

## 📊 Eğitim Sonuçları
*Gelecekte buraya modelin doğruluk (accuracy) ve kayıp (loss) grafiklerini ekleyebilirsiniz.*

## 📂 Dosya Paylaşımı ve Model
Modelin eğitilmiş `.h5` dosyasına ve diğer çalışma dosyalarına aşağıdaki bağlantıdan ulaşabilirsiniz:

🔗 **[Buraya Google Drive Linkini Yapıştır]**

---
### 🛠️ Kurulum
Bu projeyi yerel makinenizde çalıştırmak için:
1. Depoyu indirin.
2. `pip install tensorflow matplotlib tensorflow-datasets` komutuyla gerekli kütüphaneleri kurun.
3. `python cnn.py` komutuyla eğitimi başlatın.
