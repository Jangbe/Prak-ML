# %%
#Import library dan package yang dibutuhkan

import numpy as np #untuk perhitungan saintifik
import matplotlib.pyplot as plt #untuk plotting
from sklearn.linear_model import LinearRegression #import library LinearRegression dari scikit-learn

# %%
# Membuat data sembarang
penjualan = np.array([6,5,5,4,4,3,2,2,2,1])
harga = np.array([16000, 18000, 27000, 34000, 50000, 68000, 65000, 81000, 85000, 90000])

# %%
# Print data sembarang
print("Data Penjualan :", penjualan)
print("Data Harga :",harga)

# %%
plt.scatter(penjualan, harga)

#%%
#Membuat model regresi
penjualan = penjualan.reshape(-1,1) #kita tukar baris dan kolom variabel ini, agar bisa dikalikan dalam operasi matriks
#untuk lebih lengkapnya baca teori soal perhitungan regresi linier
linreg = LinearRegression()
linreg.fit(penjualan, harga)

#%%
# Visuallisasi plot hasil regresi
plt.scatter(penjualan, harga, color='red')
plt.plot(penjualan, linreg.predict(penjualan))
plt.title("Visualisasi model regresi data penjualan dan harga")
plt.xlabel("Penjualan")
plt.ylabel("Harga")

# %%
