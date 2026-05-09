# Hasil Akhir Penelitian

Folder ini berisi artefak final eksperimen penelitian dengan environment yang konsisten dengan dashboard/Streamlit.

## Ringkasan

- Topik: perbandingan `K-Means` dan `DBSCAN` untuk clustering teks pesan menfess di Telegram
- Representasi teks: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Python: `3.12.11`
- Environment package utama:
  - `fastembed==0.8.0`
  - `sentence-transformers==5.4.1`
  - `scikit-learn==1.8.0`
  - `numpy==1.26.4`
  - `pandas==3.0.2`
  - `onnxruntime==1.23.2`
  - `tokenizers==0.22.2`
  - `transformers==4.57.6`

## Struktur Folder

- `reports/`: file report hasil evaluasi akhir
- `outputs/`: file CSV hasil clustering akhir
- `input_subsets/`: subset data input yang dipakai untuk eksperimen akhir
- `config/`: script, stopword, requirements, dan metadata environment

## Definisi Skenario Data

Pembagian skenario dilakukan berdasarkan clustering pada `word_count`.

- `pendek`: `0-20` kata, pusat cluster `9.3` kata
- `menengah`: `55-128` kata, pusat cluster `77.595` kata
- `panjang`: `129-283` kata, pusat cluster `178.557` kata

## Pengaturan Eksperimen Akhir

Pengaturan umum:

- `min_len=10`
- `min_words=5`
- `limit_random=True`
- `drop_duplicate_texts=True`
- `sample_size=1000`
- `batch_size=64`
- stopword: `config/stopword.txt`

Pengaturan model:

- `K-Means`: `k=5`
- `DBSCAN`: `eps=0.22`, `min_samples=5`, `metric=cosine`, `use_umap=False`

Jumlah data yang terpakai setelah filtering dan deduplikasi:

- `pendek`: `1000 / 32564`
- `menengah`: `1000 / 1675`
- `panjang`: `300 / 314`

## Hasil Final

| Skenario | Metode | Cluster | Silhouette | Davies-Bouldin | Noise Rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Pendek | K-Means | 5 | 0.0804 | 3.8451 | 0.00% |
| Pendek | DBSCAN | 2 | 0.3186 | 1.3714 | 71.60% |
| Menengah | K-Means | 5 | 0.1403 | 3.6490 | 0.00% |
| Menengah | DBSCAN | 1 | 0.0450 | 3.3405 | 74.60% |
| Panjang | K-Means | 5 | 0.0593 | 3.4552 | 0.00% |
| Panjang | DBSCAN | 2 | 0.5492 | 1.1069 | 79.67% |

Catatan interpretasi singkat:

- `DBSCAN` memberi hasil terbaik pada skenario `pendek` dan `panjang` jika dilihat dari `Silhouette Score` dan `Davies-Bouldin Index`, tetapi menghasilkan noise yang sangat tinggi.
- `K-Means` lebih stabil karena selalu membentuk `5` cluster tanpa noise pada seluruh skenario.
- Pada skenario `menengah`, `DBSCAN` hanya membentuk `1` cluster sehingga pemisahan cluster tidak sekuat skenario lain.

## File Report Final

- `reports/kmeans_pendek_venv.txt`
- `reports/kmeans_menengah_venv.txt`
- `reports/kmeans_panjang_venv.txt`
- `reports/dbscan_pendek_venv_eps022.txt`
- `reports/dbscan_menengah_venv_eps022.txt`
- `reports/dbscan_panjang_venv_eps022.txt`

## File Output Final

- `outputs/kmeans_pendek_venv.csv`
- `outputs/kmeans_menengah_venv.csv`
- `outputs/kmeans_panjang_venv.csv`
- `outputs/dbscan_pendek_venv_eps022.csv`
- `outputs/dbscan_menengah_venv_eps022.csv`
- `outputs/dbscan_panjang_venv_eps022.csv`

## Script Acuan

- `config/kmeans_text_cluster.py`
- `config/dbscan_text_cluster.py`
- `config/length_cluster_dataset.py`
- `config/requirements.txt`
- `config/python_version.txt`
- `config/package_versions.txt`
