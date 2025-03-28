# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Load dataset
df = pd.read_csv("tips.csv")

# %%
# Hitung persentase gender
gender_counts = df['sex'].value_counts(normalize=True) * 100

# %%
# Tampilkan pie chart
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
plt.title("Persentase Pemberi Tip")
plt.show()
