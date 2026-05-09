# Lampiran Metodologi Eksperimen

## 1. Desain Eksperimen

Penelitian ini menggunakan pendekatan eksperimen komparatif untuk membandingkan dua algoritma clustering, yaitu `K-Means` dan `DBSCAN`, pada data teks pesan menfess di Telegram. Eksperimen dilakukan pada tiga skenario data berdasarkan panjang teks, yaitu `pendek`, `menengah`, dan `panjang`.

Tujuan utama eksperimen adalah:

- membandingkan performa `K-Means` dan `DBSCAN` pada data teks informal berbahasa Indonesia
- menganalisis pengaruh panjang teks terhadap kualitas cluster
- memperoleh konfigurasi hasil akhir yang konsisten dengan environment dashboard/Streamlit

## 2. Sumber Data

Data penelitian berasal dari ekspor pesan menfess Telegram dalam format CSV. File sumber utama berada pada dataset mentah project, kemudian dikelompokkan berdasarkan panjang teks menggunakan fitur `word_count`.

Subset data yang digunakan pada eksperimen akhir disimpan pada folder:

- `input_subsets/cluster_0_pendek.csv`
- `input_subsets/cluster_2_menengah.csv`
- `input_subsets/cluster_3_panjang.csv`

## 3. Definisi Skenario Data

Pembagian skenario dilakukan secara data-driven menggunakan `K-Means` pada fitur jumlah kata (`word_count`). Dari hasil pengelompokan tersebut, penelitian difokuskan pada tiga subset berikut:

- `pendek`: `0-20` kata, pusat cluster `9.3` kata, total data `32.564`
- `menengah`: `55-128` kata, pusat cluster `77.595` kata, total data `1.675`
- `panjang`: `129-283` kata, pusat cluster `178.557` kata, total data `314`

Script pembentukan cluster panjang teks tersedia pada:

- `config/length_cluster_dataset.py`

## 4. Praproses Data

Sebelum clustering, data melewati tahapan praproses yang diterapkan pada script clustering, yaitu:

- pembersihan teks
- normalisasi penulisan
- penghapusan elemen noninformatif
- penyaringan teks duplikat
- penyaringan berdasarkan panjang minimum teks

Pengaturan umum yang digunakan pada eksperimen akhir adalah:

- `min_len=10`
- `min_words=5`
- `limit=1000`
- `limit_random=True`
- `drop_duplicate_texts=True`
- `batch_size=64`
- `sample_size=1000`

Catatan:

- untuk skenario `panjang`, jumlah data yang lolos filtering dan deduplikasi adalah `300` baris dari total `314`, sehingga eksperimen akhir memakai `300` data

## 5. Representasi Teks

Setiap pesan direpresentasikan dalam bentuk embedding menggunakan model:

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Model tersebut digunakan untuk menghasilkan representasi semantik antarpesan agar kedekatan makna tidak hanya bergantung pada kesamaan kata permukaan.

## 6. Environment Eksperimen

Agar hasil konsisten dengan dashboard/Streamlit, seluruh eksperimen final dijalankan menggunakan interpreter berikut:

- `Python 3.12.11`

Interpreter dan versi package utama disimpan pada folder:

- `config/python_version.txt`
- `config/package_versions.txt`

Versi package utama:

- `fastembed==0.8.0`
- `sentence-transformers==5.4.1`
- `scikit-learn==1.8.0`
- `numpy==1.26.4`
- `pandas==3.0.2`
- `onnxruntime==1.23.2`
- `tokenizers==0.22.2`
- `transformers==4.57.6`

## 7. Konfigurasi Algoritma

Eksperimen akhir menggunakan dua algoritma dengan parameter berikut.

### 7.1 K-Means

- jumlah cluster: `k=5`

### 7.2 DBSCAN

- `eps=0.22`
- `min_samples=5`
- `metric=cosine`
- `use_umap=False`

## 8. Prosedur Eksperimen

Tahapan eksperimen akhir dilakukan sebagai berikut:

1. Menyiapkan subset data `pendek`, `menengah`, dan `panjang`.
2. Melakukan filtering berdasarkan `min_len` dan `min_words`.
3. Menghapus teks duplikat dengan opsi `drop_duplicate_texts`.
4. Mengambil sampel acak hingga `1000` data untuk tiap subset jika tersedia.
5. Mengubah teks menjadi embedding menggunakan model multibahasa yang sama untuk seluruh skenario.
6. Menjalankan `K-Means` pada setiap subset dengan `k=5`.
7. Menjalankan `DBSCAN` pada setiap subset dengan `eps=0.22` dan `min_samples=5`.
8. Menyimpan output clustering ke file CSV.
9. Menyimpan ringkasan hasil evaluasi ke file report TXT.

Perintah lengkap eksperimen akhir tersedia pada:

- `config/commands_final.txt`

## 9. Evaluasi

Evaluasi kuantitatif dilakukan menggunakan metrik berikut:

- `Silhouette Score`
- `Davies-Bouldin Index`

Khusus untuk `DBSCAN`, evaluasi juga memperhatikan:

- jumlah cluster yang terbentuk
- `noise rate`

Interpretasi umum metrik:

- nilai `Silhouette Score` yang lebih tinggi menunjukkan cluster yang lebih kompak dan lebih terpisah
- nilai `Davies-Bouldin Index` yang lebih rendah menunjukkan kualitas clustering yang lebih baik
- `noise rate` menunjukkan proporsi data yang tidak masuk ke cluster mana pun pada `DBSCAN`

## 10. Artefak Hasil Eksperimen

Report akhir disimpan pada:

- `reports/kmeans_pendek_venv.txt`
- `reports/kmeans_menengah_venv.txt`
- `reports/kmeans_panjang_venv.txt`
- `reports/dbscan_pendek_venv_eps022.txt`
- `reports/dbscan_menengah_venv_eps022.txt`
- `reports/dbscan_panjang_venv_eps022.txt`

Output cluster akhir disimpan pada:

- `outputs/kmeans_pendek_venv.csv`
- `outputs/kmeans_menengah_venv.csv`
- `outputs/kmeans_panjang_venv.csv`
- `outputs/dbscan_pendek_venv_eps022.csv`
- `outputs/dbscan_menengah_venv_eps022.csv`
- `outputs/dbscan_panjang_venv_eps022.csv`

## 11. Ringkasan Hasil Akhir

| Skenario | Metode | Data Terpakai | Cluster | Silhouette | Davies-Bouldin | Noise Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Pendek | K-Means | 1000 | 5 | 0.0804 | 3.8451 | 0.00% |
| Pendek | DBSCAN | 1000 | 2 | 0.3186 | 1.3714 | 71.60% |
| Menengah | K-Means | 1000 | 5 | 0.1403 | 3.6490 | 0.00% |
| Menengah | DBSCAN | 1000 | 1 | 0.0450 | 3.3405 | 74.60% |
| Panjang | K-Means | 300 | 5 | 0.0593 | 3.4552 | 0.00% |
| Panjang | DBSCAN | 300 | 2 | 0.5492 | 1.1069 | 79.67% |

## 12. Catatan Reproduksibilitas

Seluruh angka hasil akhir pada folder ini mengacu pada environment `.venv` yang sama dengan dashboard/Streamlit. Hasil yang dijalankan melalui interpreter atau versi package berbeda dapat menghasilkan embedding yang berbeda, sehingga memengaruhi hasil clustering, terutama pada `DBSCAN` yang sensitif terhadap perubahan jarak antardata.
