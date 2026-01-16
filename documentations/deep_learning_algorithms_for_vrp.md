# Araç Rotalama Problemleri İçin Derin Öğrenme Algoritmaları

## Özet

Bu dokümantasyon, Araç Rotalama Problemleri (Vehicle Routing Problem - VRP) için kullanılan derin öğrenme yaklaşımlarını incelemektedir. İlk olarak bu projede uygulanan model mimarisi açıklanmakta, ardından literatürdeki diğer yaklaşımlarla karşılaştırılmaktadır.

---

## 1. Giriş

Araç Rotalama Problemi (VRP), NP-zor sınıfında yer alan kombinatoryal optimizasyon problemlerinden biridir. Problem, bir veya daha fazla aracın belirli müşteri noktalarını ziyaret ederek toplam mesafeyi veya maliyeti minimize etmesini amaçlamaktadır. Geleneksel çözüm yöntemleri (tam sayı programlama, sezgisel algoritmalar) büyük ölçekli problemlerde hesaplama açısından zorlu hale gelebilmektedir.

Son yıllarda, derin öğrenme ve pekiştirmeli öğrenme tabanlı yaklaşımlar, VRP ve benzeri kombinatoryal optimizasyon problemleri için umut verici alternatifler olarak ortaya çıkmıştır. Bu yaklaşımlar, problemin yapısını öğrenerek hızlı ve kaliteli çözümler üretebilmektedir.

---

## 2. Projede Kullanılan Model Mimarisi

### 2.1 Genel Yapı

Bu projede, **Attention-Tabanlı Pointer Network** mimarisi kullanılmaktadır. Model, bir kodlayıcı-çözücü (encoder-decoder) yapısına sahiptir ve pekiştirmeli öğrenme ile eğitilmektedir.

### 2.2 Model Bileşenleri

#### 2.2.1 Kodlayıcı (Encoder)

Kodlayıcı, VRP örneğindeki her bir düğümü (depo ve müşteriler) gömme (embedding) vektörlerine dönüştürmektedir. Her düğüm üç özellik ile temsil edilmektedir:

- x koordinatı (normalize edilmiş)
- y koordinatı (normalize edilmiş)
- Talep miktarı (kapasite ile normalize edilmiş)

```python
self.embedding = nn.Linear(input_dim, hidden_dim)  # input_dim=3, hidden_dim=128
encoder_outputs = self.embedding(inputs)  # (batch, nodes, hidden)
```

#### 2.2.2 Çözücü (Decoder)

Çözücü, bir LSTM hücresi kullanarak otoregresif biçimde düğümleri sırayla seçmektedir:

```python
self.decoder_cell = nn.LSTMCell(hidden_dim, hidden_dim)
hx, cx = self.decoder_cell(decoder_input, (hx, cx))
```

#### 2.2.3 Dikkat Mekanizması (Attention Mechanism)

Model, Bahdanau tarzı toplanabilir dikkat (additive attention) mekanizmasını kullanmaktadır. Bu mekanizma, çözücünün mevcut durumunu kodlayıcı çıktıları ile ilişkilendirerek bir sonraki ziyaret edilecek düğümü seçmektedir:

```python
query = self.W_q(hx).unsqueeze(1)           # (batch, 1, hidden)
keys = self.W_k(encoder_outputs)            # (batch, nodes, hidden)
scores = self.V(torch.tanh(query + keys))   # Bahdanau attention
probs = F.softmax(scores, dim=-1)           # Düğüm seçim olasılıkları
```

### 2.3 Eğitim Yaklaşımı

Model, REINFORCE algoritması ile eğitilmektedir. Hareketli ortalama (moving average) temel çizgi (baseline) kullanılarak gradyan varyansı azaltılmaktadır:

```python
# Avantaj hesaplama
advantage = reward - moving_avg_reward

# Policy gradient kaybı
loss = (advantage * sum(tour_log_probs)).mean()
```

### 2.4 Kapasite Yönetimi

Model, öncelikle tüm müşterileri kapsayan bir TSP benzeri tur öğrenmektedir. Ardından, çıkarım sırasında bu tur, araç kapasitesine göre açgözlü bölme (greedy split) stratejisi ile ayrı rotalara ayrılmaktadır. Bu yaklaşım, VRP'nin iki aşamalı çözümünü sağlamaktadır.

---

## 3. Literatürdeki Temel Yaklaşımlar

### 3.1 Pointer Networks (Vinyals et al., 2015)

Pointer Network'ler, çıktı dizisinin giriş dizisindeki elemanlara işaret ettiği diziden diziye (sequence-to-sequence) modellerdir. Geleneksel dikkat mekanizmalarından farklı olarak, dikkat ağırlıkları doğrudan çıktı olarak kullanılmaktadır.

Bu projede kullanılan model, Pointer Network yaklaşımının temel prensiplerini uygulamaktadır.

### 3.2 REINFORCE ile Kombinatoryal Optimizasyon (Bello et al., 2016)

Bello ve arkadaşları, Pointer Network'leri pekiştirmeli öğrenme ile birleştirerek denetimli veri gerektirmeden çözüm üretmeyi başarmışlardır. REINFORCE algoritması ile model, tur uzunluğunu minimize etmeyi öğrenmektedir.

Bu projede kullanılan eğitim yaklaşımı, bu çalışmadan doğrudan esinlenmiştir.

### 3.3 Attention Model (Kool et al., 2019)

"Attention, Learn to Solve Routing Problems!" başlıklı çalışma, VRP için derin öğrenme alanında önemli bir dönüm noktası olmuştur. Model, Transformer kodlayıcısı ile çoklu kafa dikkat (multi-head attention) mekanizmasını kullanmaktadır.

**Temel Farklılıklar:**

| Bileşen     | Bu Proje                | Kool et al. (2019)                         |
| ----------- | ----------------------- | ------------------------------------------ |
| Kodlayıcı   | Linear embedding        | Transformer (Multi-Head Self-Attention)    |
| Dikkat Tipi | Toplanabilir (Additive) | Ölçekli Nokta Çarpımı (Scaled Dot-Product) |
| Baseline    | Hareketli Ortalama      | Açgözlü Yuvarlama (Greedy Rollout)         |
| Kapasite    | Çözüm sonrası bölme     | Model içinde dinamik maskeleme             |

### 3.4 DACT - Dual-Aspect Collaborative Transformer (2021)

DACT, düğüm özellikleri ve pozisyonel özellikler için ayrı gömmeler kullanarak gürültü ve uyumsuz korelasyonları azaltmaktadır. Döngüsel pozisyonel kodlama ile VRP çözümlerinin döngüsel yapısını daha iyi temsil etmektedir.

### 3.5 CaDA - Constraint-Aware Dual Attention Model (2023)

CaDA, farklı VRP varyantlarını (CVRP, VRPTW, OVRP vb.) tek bir model ile çözebilmektedir. Kısıt bilinci ekleyerek, modelin farklı problem türlerine adapte olmasını sağlamaktadır.

### 3.6 Graf Sinir Ağları (Graph Neural Networks)

GNN tabanlı yaklaşımlar, VRP'nin doğal graf yapısını doğrudan modellemektedir. Düğümler arası ilişkileri mesaj geçirme mekanizmaları ile öğrenmektedir.

---

## 4. Düğüm Sayısından Bağımsızlık

### 4.1 Teorik Perspektif

Dikkat mekanizması tabanlı modeller, teorik olarak değişken sayıda düğüm ile çalışabilmektedir. Dikkat hesaplaması, giriş boyutundan bağımsız olarak yapılabilmektedir:

```python
# Dikkat skoru hesaplama - herhangi bir num_nodes için geçerli
scores = self.V(torch.tanh(query + keys))  # (batch, num_nodes)
```

### 4.2 Pratik Sınırlamalar

Ancak pratikte önemli sınırlamalar bulunmaktadır:

1. **Eğitim Boyutu Etkisi**: Model belirli bir boyutta (örn. 30 düğüm) eğitildiğinde, farklı boyutlardaki problemlerde performans düşüşü gözlemlenmektedir.

2. **Genelleme Boşluğu**: Küçük problemlerde eğitilen modeller, büyük problemlere iyi genelleme yapamamaktadır.

### 4.3 Literatürdeki Çözümler

Bu sorunu aşmak için çeşitli yaklaşımlar önerilmiştir:

- **Çoklu Ölçek Eğitimi**: Farklı boyutlardaki problemlerle eğitim yapılması
- **Dinamik Dikkat**: Çözüm süreci boyunca düğüm gömmelerinin güncellenmesi
- **Müfredat Öğrenimi**: Küçük problemlerden büyük problemlere kademeli geçiş
- **Normalize Edilmiş Girdiler**: Koordinat ve özelliklerin tutarlı normalize edilmesi

---

## 5. Karşılaştırmalı Analiz

### 5.1 Bu Projenin Literatürdeki Konumu

```
Pointer Network (2015) ─────┐
                            ├──► Bu Proje (Hibrit Yaklaşım)
REINFORCE for CO (2016) ────┘
                                    │
                                    ▼
                        Attention Model (2019)
                                    │
                                    ▼
                    DACT (2021), CaDA (2023), GNN Tabanlı Modeller
```

### 5.2 Güçlü ve Zayıf Yönler

| Kriter             | Bu Proje | Modern Yaklaşımlar |
| ------------------ | -------- | ------------------ |
| Uygulama Basitliği | Yüksek   | Düşük              |
| Hesaplama Maliyeti | Düşük    | Yüksek             |
| Çözüm Kalitesi     | Orta     | Yüksek             |
| Genelleme Yeteneği | Sınırlı  | Güçlü              |
| Kısıt Yönetimi     | Basit    | Gelişmiş           |

---

## 6. Sonuç

Bu projede uygulanan model, Pointer Network ve REINFORCE tabanlı klasik bir yaklaşımı temsil etmektedir. Model, VRP için derin öğrenme yaklaşımlarının temel prensiplerini içermekte ve orta ölçekli problemler için kabul edilebilir çözümler üretebilmektedir.

Gelecek çalışmalarda, Transformer tabanlı kodlayıcıların entegrasyonu, daha sofistike baseline yöntemlerinin kullanımı ve kapasite kısıtlarının model içinde ele alınması gibi iyileştirmeler değerlendirilebilir.

---

## Kaynaklar

1. Vinyals, O., Fortunato, M., & Jaitly, N. (2015). Pointer Networks. _Advances in Neural Information Processing Systems_, 28.

2. Bello, I., Pham, H., Le, Q. V., Norouzi, M., & Bengio, S. (2016). Neural Combinatorial Optimization with Reinforcement Learning. _arXiv preprint arXiv:1611.09940_.

3. Kool, W., van Hoof, H., & Welling, M. (2019). Attention, Learn to Solve Routing Problems! _International Conference on Learning Representations_.

4. Ma, Y., Li, J., Cao, Z., Song, W., Zhang, L., Chen, Z., & Tang, J. (2021). Learning to Iteratively Solve Routing Problems with Dual-Aspect Collaborative Transformer. _Advances in Neural Information Processing Systems_, 34.

5. Jiang, Y., Cao, Z., Wu, Y., Song, W., & Zhang, J. (2023). Constraint-Aware Dual Attention Model for Vehicle Routing Problems. _IEEE Transactions on Neural Networks and Learning Systems_.
