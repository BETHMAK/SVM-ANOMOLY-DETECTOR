import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ✅ Load dataset (skip header row manually if needed)
df = pd.read_csv(r"c:\Users\centi\Desktop\DATASET\NLS-KDD.txt", header=None, dtype=str)

# 🔍 Remove rows where the last column has unexpected text like 'label'
df = df[df[df.columns[-1]] != 'label']

# ✅ Assign column names
column_names = [f"feature_{i}" for i in range(df.shape[1])]
df.columns = column_names

# ✅ Rename the last column to "attack_label" for clarity
attack_column = column_names[-1]
df.rename(columns={attack_column: "attack_label"}, inplace=True)
attack_column = "attack_label"

# ✅ Debug: check original values
print("Unique labels before processing:", df[attack_column].unique())

# ✅ Define NORMAL labels (adjust according to your dataset documentation)
NORMAL_LABELS = ["normal", "0", "1"]  # Use only the *true normal* labels

# ✅ Map labels: 0 = normal, 1 = attack
df[attack_column] = df[attack_column].apply(lambda x: 0 if x.strip().lower() in NORMAL_LABELS else 1)

# ✅ Check label distribution
print("\nLabel distribution after processing:\n", df[attack_column].value_counts())

# ✅ Split categorical and numerical features
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
numerical_columns = df.select_dtypes(exclude=['object']).columns.tolist()

# ✅ Remove label column from categorical
if attack_column in categorical_columns:
    categorical_columns.remove(attack_column)

# ✅ Encode categoricals
encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

# ✅ Normalize numerical columns
scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# ✅ Save processed data
save_dir = r"C:\Users\centi\Desktop\DATASET\SVM_Anomaly_Detection"
os.makedirs(save_dir, exist_ok=True)

processed_file_path = os.path.join(save_dir, "processed_dataset.csv")
df.drop(columns=[attack_column]).to_csv(processed_file_path, index=False)

labels_path = os.path.join(save_dir, "labels.csv")
df[[attack_column]].to_csv(labels_path, index=False)

print("✅ Data processing complete!")
print("📂 Processed dataset saved at:", processed_file_path)
print("📂 Labels saved at:", labels_path)
