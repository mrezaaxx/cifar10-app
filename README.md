---
title: Group 1 Cifar10 App
emoji: 🖼️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
short_description: Klasifikasi gambar CIFAR-10 dengan CNN (Streamlit)
pinned: false
---

# Klasifikasi Gambar CIFAR-10 (Tugas Kelompok 2)

Aplikasi web Streamlit yang menerima gambar (JPG/PNG) dan menampilkan prediksi
model CNN untuk 10 kelas CIFAR-10: airplane, automobile, bird, cat, deer, dog,
frog, horse, ship, truck.

Fitur:
- Unggah gambar, tampilkan gambar asli dan versi 32×32 yang dilihat model.
- Prediksi top-1 + confidence, top-3 dalam bar chart.
- Peringatan bila confidence < 50%.
- Penanganan galat: format salah, file kosong/rusak, file > 5 MB, model gagal dimuat.

## Struktur

```
app.py                  # antarmuka Streamlit
inference.py            # preprocessing + prediksi (dipakai app & skrip evaluasi)
config.json             # kontrak model: kelas, ukuran input, normalisasi
model/model.keras       # file model
train_temp.py           # pelatihan model sementara
evaluate_new_images.py  # uji gambar baru -> results.csv
test_images/<kelas>/    # gambar uji di luar dataset
requirements.txt
```

## Menjalankan secara lokal

Butuh **Python 3.11** (TensorFlow 2.18 belum mendukung Python 3.13+).

```bash
git clone <url-repo> cifar10-app
cd cifar10-app
python -m venv .venv            # pastikan python = 3.11
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Buka http://localhost:8501.

## Mengganti dengan model final tim

1. Simpan model final sebagai `model/model.keras` (atau `.h5`, lalu ubah `model_path`).
2. Sesuaikan `config.json`:
   - `class_names` — urutan **harus sama** dengan urutan output model.
   - `input_size` — ukuran input model, mis. `[32, 32]`.
   - `normalization` — saat ini `rescale_255` (piksel ÷ 255). Jika model memakai
     normalisasi lain (mis. preprocess_input untuk transfer learning), tambahkan di
     `inference.preprocess`.
3. Pastikan versi TensorFlow sama dengan yang dipakai saat training.

Normalisasi atau urutan kelas yang berbeda tidak memunculkan error, tetapi
membuat prediksi salah secara diam-diam.

## Uji gambar baru

Taruh gambar di `test_images/<nama_kelas>/`, contoh `test_images/cat/kucing1.jpg`.
Gambar di luar 10 kelas boleh ditaruh di `test_images/unknown/`.

```bash
python evaluate_new_images.py
```

Hasil per gambar ditulis ke `results.csv` untuk bab evaluasi laporan.

## Deployment ke Hugging Face Spaces

URL publik (Streamlit Community Cloud): https://group-1-cifar10.streamlit.app/

Repo kode: https://github.com/mrezaaxx/cifar10-app

Space Hugging Face (kode sama, saat ini dijeda karena kuota CPU akun 0):
https://huggingface.co/spaces/mrezaaxx/group-1-cifar10-app

Hugging Face tidak lagi menyediakan SDK Streamlit bawaan, sehingga aplikasi
dijalankan sebagai **Docker Space** (`sdk: docker`). `Dockerfile` memasang
Python 3.11 + `requirements.txt`, lalu menjalankan `streamlit run app.py` di port 7860.

1. Buat Space baru di https://huggingface.co/new-space, SDK: **Docker** (Blank).
2. Login dan push:
   ```bash
   pip install huggingface_hub
   hf auth login --add-to-git-credential
   git lfs install
   git init -b main
   git remote add origin https://huggingface.co/spaces/<user>/<nama-space>
   git add .
   git commit -m "Initial app"
   git push -u origin main
   ```
   `.gitattributes` sudah mengatur `*.keras` dan `*.h5` lewat Git LFS.
3. Tunggu status build **Running**, lalu uji URL publik.
4. Space gratis akan tidur bila lama tidak dipakai. Buka URL beberapa menit sebelum demo.

Cadangan: Streamlit Community Cloud — push repo ke GitHub, lalu deploy dari
https://share.streamlit.io dengan file utama `app.py` dan Python 3.11.

## Keterbatasan

- Model hanya mengenal 10 kelas; gambar di luar kelas tetap dipaksa ke salah satunya.
- Input diperkecil ke 32×32, sehingga detail gambar beresolusi tinggi hilang.
- Confidence softmax tinggi bukan jaminan prediksi benar.
