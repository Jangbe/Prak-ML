# %%
# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree

# %%
# Load dataset
df = pd.read_csv("citrus.csv")
print(df.head())
print(df.describe())
print(df['name'].value_counts())

# %%
# Encode label buah
df['name'] = df['name'].map({'orange': 0, 'grapefruit': 1})

# %%
# Visualisasi fitur
sns.pairplot(df, hue='name')
plt.show()

# %%
# Pisahkan fitur dan label
X = df.drop('name', axis=1)
y = df['name']

# %%
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Data training: {len(X_train)}, Data testing: {len(X_test)}")

# %%
# Latih model Decision Tree
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)

# %%
# Prediksi dan evaluasi
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# %%
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix')
plt.show()

# %%
# Visualisasi pohon keputusan
fig, ax = plt.subplots(figsize=(12, 8))
tree.plot_tree(model, feature_names=X.columns, class_names=['orange', 'grapefruit'], filled=True)
plt.show()

# %%
# Prediksi data baru
sample = pd.DataFrame([[6.3, 130.3, 246, 116, 56]], columns=X.columns)
prediction = model.predict(sample)
print("Prediksi buah:", "orange" if prediction[0] == 0 else "grapefruit")

# %%
