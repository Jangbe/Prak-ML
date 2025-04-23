# Decision Tree Classifier: Oranges vs Grapefruit

## Tujuan:
Membangun model klasifikasi untuk membedakan antara buah jeruk (orange) dan anggur (grapefruit) menggunakan algoritma Decision Tree.

## Tahapan Pembuatan Model
1. Import Library <br>
Import pustaka yang dibutuhkan: pandas, numpy, matplotlib, seaborn, dan pustaka machine learning dari sklearn.

2. Load dan Eksplorasi Data
    * Buka file citrus.csv
    * Lihat bentuk data dan cek apakah terdapat missing value
    * Tampilkan deskripsi statistik dasar

3. Visualisasi Data
    * Gunakan pairplot() untuk melihat persebaran fitur
    * Lihat distribusi label (jeruk vs anggur)

4. Preprocessing
    * Encode label kategorikal (fruit) menjadi angka
    * Cek skala data dan pastikan tipe data sesuai

5. Split Data
    * Pisahkan data menjadi data latih dan data uji (contoh: 70% train, 30% test)

6. Latih Model Decision Tree
    * Gunakan DecisionTreeClassifier(criterion='entropy')
    * Fit model dengan data latih

7. Evaluasi Model
    * Tampilkan classification report (precision, recall, f1-score, accuracy)
    * Buat confusion matrix
    * Visualisasikan Decision Tree

8. Uji Prediksi
    * Masukkan input baru dan prediksi apakah buah tersebut jeruk atau anggur