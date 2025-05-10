# Analisis Sentimen dengan SVM

## Deskripsi Proyek

Proyek ini bertujuan untuk melakukan **Analisis Sentimen** menggunakan algoritma **Support Vector Machine (SVM)** untuk klasifikasi teks. Model ini akan mengelompokkan teks menjadi **sentimen positif** atau **sentimen negatif** berdasarkan dataset yang diberikan. Proyek ini mengikuti panduan yang diberikan pada artikel Medium tentang Sentiment Analysis menggunakan SVM.

## Dataset

Dataset yang digunakan adalah dataset ulasan (reviews) atau data teks lain yang memiliki label sentimen (positif atau negatif). Dataset ini bisa berupa dataset Twitter, ulasan produk, atau komentar sosial media.

### Struktur Dataset

| Fitur | Deskripsi                                   |
| :---- | :------------------------------------------ |
| text  | Teks atau ulasan yang akan diklasifikasikan |
| label | Label sentimen (misal: positif/negatif)     |

## Alur Kerja

1. **Memuat Data:** Membaca dataset ke dalam DataFrame pandas.
2. **Pra-pemrosesan Data:** Membersihkan teks (misalnya menghapus stopwords, mengubah huruf menjadi kecil).
3. **Ekstraksi Fitur:** Menggunakan TF-IDF untuk mengonversi teks ke bentuk numerik.
4. **Pelatihan Model:** Melatih model SVM dengan data yang sudah diproses.
5. **Evaluasi Model:** Mengevaluasi performa model menggunakan classification report dan confusion matrix.
6. **Prediksi:** Menguji model dengan sampel teks baru.
