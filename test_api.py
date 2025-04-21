import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json

# ✅ Load feature data and labels
data_path = r"C:\Users\centi\Desktop\DATASET\SVM_Anomaly_Detection\processed_dataset.csv"
labels_path = r"C:\Users\centi\Desktop\DATASET\SVM_Anomaly_Detection\labels.csv"

X = pd.read_csv(data_path)
y = pd.read_csv(labels_path)

# ✅ Combine features and labels for analysis and testing
df = pd.concat([X, y], axis=1)
df.rename(columns={df.columns[-1]: "Label"}, inplace=True)

# ✅ Plot class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="Label", data=df, palette="Set2")
plt.title("Распределение нормальных и Атака Трафика")
plt.xlabel("Этикетка (0 = Обычный, 1 = Атака )")
plt.ylabel("Count")
plt.xticks([0, 1], ["Обычный", "Атака"])
plt.tight_layout()
plt.show()

