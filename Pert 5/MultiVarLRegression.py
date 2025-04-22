#%%
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
# %%
df = pd.read_csv('data.csv')
df
# %%
import math
median_bedroom = math.floor(df.bedrooms.median())
median_bedroom
# %%
df.bedrooms = df.bedrooms.fillna(median_bedroom)
df
# %%
reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', 'age']], df.price)
# %%
reg.coef_
# %%
reg.intercept_
# %%
reg.predict([[3000, 3, 40]])
# %%
