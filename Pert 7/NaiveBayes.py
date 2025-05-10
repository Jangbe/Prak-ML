#%%
# Loading library python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# %%
# Read dataset
df_net = pd.read_csv('Social_Network_Ads.csv')
df_net.head()

# %%
# Preprocessing - Get required data
df_net.drop(columns = ['User ID'], inplace=True)
df_net.head()

# %%
# Describe data
df_net.describe()

# %%
# Label encoding
le = LabelEncoder()
df_net['Gender']= le.fit_transform(df_net['Gender'])

# %%
# Correlation matrix
df_net.corr()

# %%
sns.heatmap(df_net.corr())

# %%
# Drop Gender column
df_net.drop(columns=['Gender'], inplace=True)

# %%
sns.heatmap(df_net.corr())

# %%
# Relationship between Age and Salary
plt.scatter(df_net['Age'], df_net['EstimatedSalary'])

# %%
# Split data into independent/dependent variables
X = df_net.iloc[:, :-1].values
y = df_net.iloc[:, -1].values

# %%
# Split data into Train/Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = True)

# %%
# Scale dataset
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%%
# Print length of train and test dataset
print(len(X_train))
print(len(X_test))

# %%
# Train Bayes-Theorem model
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# %%
# Prediction
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# %%
# Accuracy
accuracy_score(y_test, y_pred)

# %%
# Classification report
print(f'Classification Report: \n{classification_report(y_test, y_pred)}')

# %%
# F1 score
print(f"F1 Score : {f1_score(y_test, y_pred)}")

# %%
# Confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)

# %%
# Plot Precision-Recall Curve
y_pred_proba = classifier.predict_proba(X_test)[:,1]
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(recall, precision, label='Naive Bayes')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend()

# Plot AUC/ROC curve
y_pred_proba = classifier.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test,  y_pred_proba)

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(fpr, tpr, label='Naive Bayes Classification', color = 'firebrick')
ax.set_title('ROC Curve')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
plt.box(False)
ax.legend()

# %%
# Predict purchase with Age(30) and Salary(87000)
print(classifier.predict(sc.transform([[30, 87000]])))
# %%

