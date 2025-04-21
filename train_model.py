import pandas as pd
from sklearn.svm import OneClassSVM
import joblib

# ✅ Load preprocessed data
X = pd.read_csv(r"C:\Users\centi\Desktop\DATASET\SVM_Anomaly_Detection\processed_dataset.csv")
y = pd.read_csv(r"C:\Users\centi\Desktop\DATASET\SVM_Anomaly_Detection\labels.csv").values.ravel()

# ✅ Check class distribution
print("Class Distribution:", pd.Series(y).value_counts())

# ✅ Filter out the attack class (1.0) - keep only normal class (0.0)
X_filtered = X[y != 1.0]  # Keep only normal class (0.0)
y_filtered = y[y != 1.0]  # Remove attack class labels

# ✅ Check new class distribution
print("\nClass Distribution after removing attack class:", pd.Series(y_filtered).value_counts())

# ✅ Anomaly detection with OneClassSVM (since we're training on only normal class)
ocsvm = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
ocsvm.fit(X_filtered)  # Only normal data is used

# ✅ Save the trained OneClassSVM model
model_path = r"C:\Users\centi\Desktop\DATASET\SVM_Anomaly_Detection\ocsvm_model.pkl"
joblib.dump(ocsvm, model_path)

print("✅ Anomaly Detection model training complete with normal class only!")
print("📂 Saved as:", model_path)
