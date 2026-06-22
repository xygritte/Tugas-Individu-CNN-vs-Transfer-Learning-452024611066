## Analisis Komprehensif

### 1. Evaluasi Dataset

* **Kapasitas Dataset untuk CNN Independen:** Penggunaan 700 sampel *training* dari dataset CIFAR-10 sangat sub-optimal untuk melatih CNN secara mandiri (*from scratch*). Idealnya, arsitektur CNN membutuhkan setidaknya 5.000–10.000 citra per kelas. Keterbatasan data pada skala ini secara signifikan meningkatkan risiko *overfitting*.
* **Variasi Visual dan Resolusi:** Meskipun CIFAR-10 memiliki resolusi spasial yang rendah (32×32 piksel) yang membatasi detail tekstur, dataset ini tetap menawarkan variasi sudut pandang, pencahayaan, dan latar belakang yang esensial untuk mendukung kemampuan generalisasi model dasar.
* **Keseimbangan Distribusi Kelas:** Kategori *Airplane* dan *Automobile* terdistribusi secara proporsional. Skema partisi kelas yang seimbang (rasio ~50:50) ini memastikan model tidak mengalami bias mayoritas kelas selama proses pembelajaran.
* **Kompleksitas Citra:** Resolusi rendah pada CIFAR-10 menghasilkan tingkat derau (*noise*) yang minim, namun berdampak pada hilangnya fitur granular. Sebagai perbandingan, dataset *Cats vs Dogs* menghadirkan kompleksitas visual yang lebih tinggi melalui rentang pose dan latar belakang yang dinamis.
* **Korelasi Kualitas Data terhadap Arsitektur:** Arsitektur CNN mandiri memiliki sensitivitas absolut terhadap kuantitas data. Pada volume sampel yang sangat minim, model cenderung mengalami *underfitting* atau *overfitting*. Sebaliknya, metode *Transfer Learning* terbukti lebih resilien dan adaptif terhadap kelangkaan data karena mengeksploitasi kumpulan fitur visual yang telah diekstraksi sebelumnya.

---

### 2. Evaluasi Kinerja Komparatif Model

* **Superioritas Performa:** Implementasi *Transfer Learning* (MobileNetV2) secara konsisten mencapai tingkat akurasi *testing* yang lebih superior dalam siklus konvergensi yang lebih efisien (5 vs 10 *epoch*). Hal ini memvalidasi keunggulan fitur prabangun (*pretrained features*) untuk skenario data terbatas.
* **Validasi Metrik Akurasi:** Tingkat akurasi global dapat menghasilkan interpretasi yang menyesatkan, terutama pada kasus bias dataset atau indikasi *overfitting*. Penggunaan *Confusion Matrix* dan *Classification Report* (*Precision, Recall, F1-Score*) menawarkan parameter evaluasi kinerja yang jauh lebih faktual.
* **Indikator Overfitting:** Model CNN mandiri memperlihatkan margin diskrepansi yang signifikan antara akurasi *training* dan *validation*, yang merupakan manifestasi kuat dari *overfitting*. Risiko ini berhasil ditekan pada *Transfer Learning* melalui strategi pembekuan lapisan parameter (*layer freezing*).
* **Stabilitas Kurva Pelatihan:** Ketiadaan pembaruan bobot pada lapisan ekstraksi fitur utama membuat trajektori akurasi validasi pada *Transfer Learning* menjadi sangat stabil. Sebaliknya, pembaruan seluruh parameter komputasi pada CNN mandiri di setiap *epoch* menghasilkan grafik fungsi *loss* yang fluktuatif.
* **Sensitivitas Skala Data:** Ketergantungan struktural CNN mandiri terhadap ukuran dataset sangat tinggi. Di sisi lain, *Transfer Learning* mampu mereplikasi performa optimal walau dengan suplai data marginal, karena mewarisi basis pengetahuan hierarkis dari ImageNet (yang dilatih menggunakan 1,2 juta citra).

---

### 3. Matriks Pemilihan Pendekatan Algoritme

| Parameter Penentu | Rekomendasi: CNN *From Scratch* | Rekomendasi: *Transfer Learning* |
| --- | --- | --- |
| **Kuantitas Dataset** | Skala masif (>100.000 citra per kelas) | Skala minor hingga menengah (<50.000 citra) |
| **Karakteristik Domain** | Sangat spesifik / anomali visual ekstrem (mis. citra radiologi, inframerah satelit) | Memiliki kesamaan struktural dengan ImageNet (objek generik, hewan) |
| **Kapasitas Komputasi** | Infrastruktur tinggi (GPU/TPU *Cluster*) dengan durasi *training* tidak terbatas | Terbatas, mengutamakan efisiensi daya komputasi dan memori |
| **Orientasi Proyek** | Modifikasi arsitektur penuh, penyebaran (*deployment*) komputasi *Edge* | Penyusunan purwarupa (*prototyping*) kilat dengan target presisi tinggi |

---

### 4. Studi Kasus dan Justifikasi Pengambilan Keputusan

* **Skenario 1: Diagnostik Medis Berbasis 300 Citra**
* **Pilihan:** *Transfer Learning (Feature Extraction)*
* **Justifikasi:** Pelatihan CNN mandiri pada skala ini dipastikan akan memicu kegagalan ekstraksi pola (*overfitting* parah). Membekukan parameter model dasar (seperti ResNet/MobileNet) memastikan pengklasifikasi baru dilatih secara efisien. Penyesuaian akhir (*fine-tuning*) dapat diaplikasikan pada lapisan terdalam untuk adaptasi visual medis yang spesifik.


* **Skenario 2: Katalog 1 Juta Citra Produk Internal Perusahaan**
* **Pilihan:** *Transfer Learning + Full Fine-Tuning* (Pendekatan Hibrida)
* **Justifikasi:** Walaupun skala data memenuhi syarat kelayakan CNN mandiri, menginisiasi pemodelan dari bobot awal *pretrained model* terbukti jauh lebih efisien untuk memangkas durasi konvergensi gradien secara drastis dibandingkan mengandalkan inisialisasi bobot acak.


* **Skenario 3: Proyek Purwarupa Berdurasi 2 Hari (500 Citra)**
* **Pilihan:** *Transfer Learning (Feature Extraction)*
* **Justifikasi:** Limitasi jadwal secara otomatis menganulir opsi riset arsitektur maupun siklus *hyperparameter tuning* yang panjang. MobileNetV2 dengan pengaturan pengklasifikasi sederhana mampu mencapai metrik kelayakan hanya dalam hitungan menit.


* **Skenario 4: Domain Ekstrem Spesifik dengan Dukungan GPU Masif**
* **Pilihan:** *Transfer Learning + Full Fine-Tuning*
* **Justifikasi:** Paradigma *State-of-The-Art* (SOTA) saat ini mengonfirmasi bahwa melonggarkan seluruh parameter (*unfreeze all layers*) pada arsitektur prabangun tetap menghasilkan konvergensi 3 hingga 10 kali lipat lebih cepat ketimbang membangun ulang arsitektur, terlepas dari keunikan domain datanya.



---

## Refleksi Pribadi

* **Hambatan Eksekusi Teknis:**
Tantangan paling menyita waktu adalah merancang fungsi *pipeline* `tf.data` yang selaras dengan spesifikasi *preprocessing Transfer Learning*. Implementasi MobileNetV2 mewajibkan rentang normalisasi [-1, 1], yang mana sangat kontras dengan format standar [0, 1] pada CNN konvensional. Kesalahan implementasi pada tahap ini berjalan senyap (tanpa *error code* eksplisit) namun berimplikasi destruktif terhadap perhitungan akurasi.
* **Kompleksitas Pemahaman Konseptual:**
Arsitektur *Transfer Learning* menuntut transisi pemahaman teoretis yang tajam. Dibutuhkan kemampuan analitis untuk memetakan arsitektur lapis prabangun, menetapkan titik pembekuan (*freezing layer*) secara presisi, serta memanipulasi *input shape* dan fungsi *preprocessing* bawaan.
* **Dikotomi Paradigma Pelatihan:**
Pembuatan CNN mandiri terasa layaknya proses manufaktur dari titik nol—di mana intervensi struktural diperlukan di setiap lapis arsitektur. Sebaliknya, metode *Transfer Learning* dapat diibaratkan sebagai proses daur ulang intelijen visual; kita mendelegasikan beban komputasi masif yang telah dieksekusi di masa lampau untuk mendapatkan akselerasi hasil yang instan.
* **Preferensi Praktik Industri:**
Penerapan *Transfer Learning* akan menjadi prioritas operasional saya dalam penyelesaian masalah nyata. Skalabilitas, stabilitas pelatihan, dan efisiensi durasi hingga masuk ke fase siklus penyebaran sistem (*deployment*) menjadikannya opsi yang paling rasional untuk diimplementasikan.
* **Sintesis Pengetahuan:**
Wawasan krusial dari eksperimen ini mematahkan asumsi sentris bahwa metrik akurasi adalah satu-satunya acuan keberhasilan *machine learning*. Perancangan arsitektur sejatinya merupakan kalkulasi *trade-off* multidimensional antara aset data pendukung, spesifikasi perangkat keras keras komputasi, limitasi kalender proyek, serta metodologi mitigasi *overfitting*. Parameter dan batasan proyek adalah penentu mutlak algoritme apa yang pantas dikerahkan.
