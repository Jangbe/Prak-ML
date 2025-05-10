# Pima Indians Diabetes - Naive Bayes Classifier

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# %%
# Load dataset
df = pd.read_csv('diabetes.csv')
print(df.head())
print(df.describe())
print(df.info())

# %%
# Check for missing values
print(df.isnull().sum())

# %%
# Split data into features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# %%
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %%
# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# %%
# Evaluate model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# %%
# Test with new sample
sample = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])  # contoh data baru
prediction = model.predict(sample)
print("Prediction for new sample:", "Diabetes" if prediction[0] == 1 else "No Diabetes")
