#%% 
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
# %%
df = pd.read_csv("https://raw.githubusercontent.com/sintya1234/datasets/main/winequality-red.csv")
df
# %%
df.describe()
# %%
plt.figure(figsize=[19,10],facecolor='pink')
sns.heatmap(df.corr(),annot=True)
# %%
sns.boxplot(data=df)
plt.show()
# %%
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# %%
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
# %%
import statsmodels.api as sm
X = sm.add_constant(X_train)
model = sm.OLS(y_train, X).fit()
print(model.summary())
# %%
y_pred = reg.predict(X_test)
evaluate = pd.DataFrame({"y_test":y_test, "y_pred":y_pred})
evaluate.head()
# %%
from sklearn.metrics import mean_squared_error
print("Root Mean Squared Error (RMSE): ", np.sqrt(mean_squared_error(y_test, y_pred)))
# %%
