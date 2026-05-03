# Text Clustering Dashboard

Project ini dipakai untuk:
- grab data Telegram ke CSV
- clustering teks dengan KMeans dan DBSCAN berbasis Sentence-BERT-compatible embedding
- menjalankan proses dari dashboard web

Data utama ada di `data/raw/export.csv` dengan kolom yang disimpan hanya:
- `id`
- `date`
- `text`

## Struktur Project

- `dashboard.py`: dashboard web utama
- `scripts/tg_grab_csv.py`: grab data Telegram ke CSV
- `scripts/kmeans_text_cluster.py`: clustering dengan KMeans
- `scripts/dbscan_text_cluster.py`: clustering dengan DBSCAN
- `scripts/run_all_clusters.py`: jalankan KMeans dan DBSCAN sekaligus
- `data/raw/`: data mentah
- `data/sample/`: sample data kecil
- `data/processed/`: hasil clustering CSV
- `reports/`: report hasil clustering
- `resources/`: stopword dan resource pendukung
- `archive/`: hasil eksperimen lama

## Setup

Install dependency:

```bash
pip install -r requirements.txt
```

Kalau pakai virtualenv lokal project:

```bash
source .venv/bin/activate
```

## Dashboard Web

Jalankan dashboard:

```bash
streamlit run dashboard.py
```

Kalau tidak activate virtualenv:

```bash
.venv/bin/streamlit run dashboard.py
```

Fitur dashboard:
- pilih file input CSV
- pilih kolom teks
- ubah parameter umum, KMeans, dan DBSCAN
- jalankan KMeans saja, DBSCAN saja, atau dua-duanya
- lihat command yang dipakai
- lihat log proses
- lihat preview hasil CSV
- lihat grafik distribusi cluster
- lihat isi report

## Grab Data Telegram

Script grab Telegram sekarang hanya menyimpan kolom:
- `id`
- `date`
- `text`

Contoh:

```bash
python3 scripts/tg_grab_csv.py \
  --channel @nama_channel \
  --out data/raw/export.csv
```

Argumen yang sering dipakai:
- `--channel`: username atau URL channel Telegram
- `--out`: output CSV
- `--resume-from-id`: lanjut grab dari ID tertentu
- `--reverse`: ambil dari post lama ke baru
- `--limit`: batasi jumlah message
- `--progress-every`: interval progress log

Credential bisa diisi lewat `.env`:

```env
TG_API_ID=...
TG_API_HASH=...
TG_SESSION=tg_grab
```

## Input Data

Input default clustering adalah `data/raw/export.csv`.

Syarat minimal:
- harus ada kolom `text`

Kolom `id` dan `date` opsional untuk analisis, tapi tetap dipertahankan di project ini.

## Preprocessing

KMeans dan DBSCAN memakai preprocessing dasar yang sama:
- lowercase
- normalisasi whitespace
- hapus URL
- hapus mention seperti `@user`
- hapus hashtag
- hapus boilerplate umum seperti `sansfess`, `kirimin aku pesan rahasia`, `pesan rahasia`
- normalisasi kata informal tertentu

Stopword default dibaca dari `resources/stopword.txt`.

Catatan:
- stopword dipakai untuk keyword report, bukan untuk embedding utama
- `min-len` dan `min-words` dipakai untuk membuang teks yang terlalu pendek setelah cleaning

## Jalankan Dua Model Sekaligus

Cara paling praktis:

```bash
python3 scripts/run_all_clusters.py
```

Contoh:

```bash
python3 scripts/run_all_clusters.py \
  --input data/raw/export.csv \
  --k 10 \
  --eps 0.7 \
  --min-samples 10 \
  --drop-duplicate-texts
```

Untuk cek command tanpa mengeksekusi:

```bash
python3 scripts/run_all_clusters.py --dry-run
```

## Jalankan KMeans

Contoh default:

```bash
python3 scripts/kmeans_text_cluster.py \
  --input data/raw/export.csv \
  --text-col text \
  --k 10 \
  --output-csv data/processed/export_clustered.csv \
  --report reports/cluster_report.txt
```

Parameter yang paling sering dipakai:
- `--k`: jumlah cluster
- `--limit`: batasi jumlah data
- `--limit-random`: sampling acak saat pakai `--limit`
- `--drop-duplicate-texts`: hapus teks duplikat setelah cleaning
- `--filter-passes`: jumlah pass filter cluster kecil
- `--min-cluster-size`: threshold cluster kecil saat filter pass
- `--embedding-model`: model embedding
- `--batch-size`: batch size embedding
- `--device`: `cpu`, `mps`, atau `cuda`
- `--stopwords`: stopword tambahan untuk report
- `--no-default-stopwords`: matikan stopword default
- `--sample-size`: jumlah sampel untuk metric evaluasi

Parameter report keyword:
- `--max-features`
- `--min-df`
- `--max-df`
- `--ngram-max`

Auto stopword opsional:
- `--auto-stopwords`
- `--auto-stopwords-min-df-ratio`
- `--auto-stopwords-out`

Contoh filtering cluster kecil:

```bash
python3 scripts/kmeans_text_cluster.py \
  --input data/raw/export.csv \
  --text-col text \
  --k 10 \
  --drop-duplicate-texts \
  --filter-passes 1 \
  --min-cluster-size 5 \
  --output-csv data/processed/export_clustered.csv \
  --report reports/cluster_report.txt
```

Output utama KMeans:
- `data/processed/export_clustered.csv`
- `reports/cluster_report.txt`

Isi hasil CSV biasanya mencakup:
- kolom asli input
- `text_clean`
- `cluster`
- `filtered_out`
- `filtered_pass`

## Jalankan DBSCAN

Contoh default:

```bash
python3 scripts/dbscan_text_cluster.py \
  --input data/raw/export.csv \
  --text-col text \
  --eps 0.7 \
  --min-samples 10 \
  --output-csv data/processed/export_dbscan_clustered.csv \
  --report reports/dbscan_cluster_report.txt
```

Parameter yang paling sering dipakai:
- `--eps`: radius cosine distance
- `--min-samples`: minimal tetangga untuk core point
- `--limit`: batasi jumlah data
- `--limit-random`: sampling acak saat pakai `--limit`
- `--drop-duplicate-texts`: hapus teks duplikat setelah cleaning
- `--embedding-model`: model embedding
- `--batch-size`: batch size embedding
- `--device`: `cpu`, `mps`, atau `cuda`
- `--sample-size`: jumlah sampel untuk metric evaluasi

Tuning `eps`:

```bash
python3 scripts/dbscan_text_cluster.py \
  --input data/raw/export.csv \
  --text-col text \
  --limit 5000 \
  --min-samples 10 \
  --eps-scan 0.55,0.6,0.65,0.7,0.75,0.8
```

Output utama DBSCAN:
- `data/processed/export_dbscan_clustered.csv`
- `reports/dbscan_cluster_report.txt`

## Output dan Metric

Hasil clustering disimpan sebagai:
- CSV hasil labeling cluster
- report text berisi metric dan contoh isi cluster

Metric yang biasa muncul di report:
- `silhouette_cosine`
- `davies_bouldin`
- `calinski_harabasz`
- `stability_ari_*` untuk KMeans
- `noise_rate` untuk DBSCAN

Catatan penting:
- `sample-size` hanya memengaruhi perhitungan metric evaluasi, bukan hasil clustering
- untuk dataset besar, metric bisa dihitung dari sampel agar lebih cepat

## Catatan Praktis

- Untuk teks, silhouette yang kecil masih bisa wajar
- Fokus utama tetap pada interpretasi cluster, top terms, dan contoh teks
- Jika keyword masih terlalu generik, tambahkan stopword di `resources/stopword.txt`
- Untuk DBSCAN pada data besar, tuning `eps` di subset dulu lalu baru jalankan full
