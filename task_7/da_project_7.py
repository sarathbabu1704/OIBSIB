import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import IsolationForest

df = pd.read_csv("creditcard.csv")
print("First 5 rows of dataset:")
print(df.head())
print("\nDataset Information:")
print(df.info())
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42
)

df['anomaly_score'] = iso_forest.fit_predict(
    df.drop('Class', axis=1)
)

# Convert Isolation Forest output
df['anomaly'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)

print("\nAnomaly Detection Results:")
print(df['anomaly'].value_counts())
# Log transform Amount (important for fraud detection)
df['Amount_log'] = np.log1p(df['Amount'])

# Final features and target
X = df.drop(['Class', 'anomaly_score'], axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))
dt = DecisionTreeClassifier(
    max_depth=6,
    random_state=42
)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("\n--- Decision Tree ---")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))
nn = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=300,
    random_state=42
)
nn.fit(X_train, y_train)
nn_pred = nn.predict(X_test)

print("\n--- Neural Network ---")
print("Accuracy:", accuracy_score(y_test, nn_pred))
print(classification_report(y_test, nn_pred))
models = {
    "Logistic Regression": lr_pred,
    "Decision Tree": dt_pred,
    "Neural Network": nn_pred
}

for name, pred in models.items():
    plt.figure()
    sns.heatmap(
        confusion_matrix(y_test, pred),
        annot=True,
        fmt='d',
        cmap='Reds'
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
def real_time_fraud_detection(transaction, model):
    transaction_scaled = scaler.transform([transaction])
    prediction = model.predict(transaction_scaled)

    if prediction[0] == 1:
        print("🚨 FRAUD DETECTED!")
    else:
        print("✅ Transaction is SAFE")

# Simulate incoming transaction
sample_transaction = X.iloc[0].values
real_time_fraud_detection(sample_transaction, lr)
def batch_fraud_detection(data, model):
    predictions = model.predict(data)
    fraud_count = np.sum(predictions)
    print(f"\nTotal Fraudulent Transactions Detected: {fraud_count}")

batch_fraud_detection(X_test, lr)