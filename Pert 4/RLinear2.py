#%%
# import library dan package yang dibutuhkan
import pandas as pd #untuk dataframe
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression #import library LinearRegression dari scikit-learn

# %%
#Membaca data CSV
df = pd.read_csv("FuelConsumptionCo2.csv") #membaca data

# melihat 5 baris pertama data
df.head()

#%%
#kita ambil kolom mana saja yang akan kita analisis, dan membuang sisanya
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','CO2EMISSIONS']]
cdf.head(9)

# %%
#Kita plot hubungannya
plt.scatter(cdf.FUELCONSUMPTION_CITY, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_CITY")
plt.ylabel("Emission")
plt.show()

# %%
#Kita plot hubungannya
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# %%
#Membagi data
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# %%
#Visualiasi data training antara engine size dan emission
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# %%
#Membuat model regresi
regr = LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

# Koefisien model
print('Coefficients: ', regr.coef_)
print('Intercept: ',regr.intercept_)

# %%
#Plot hasil regresi
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

# %%
result = regr.predict([[3.5]])
print("Prediksi emisi untuk engine size 3.5 adalah", result[0][0])
# %%
