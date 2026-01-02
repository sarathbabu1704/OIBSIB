import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
df = pd.read_csv("WineQT.csv")
print("First 5 rows of dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Distribution of wine quality
plt.figure()
sns.countplot(x='quality', data=df)
plt.title("Wine Quality Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap of Chemical Properties")
plt.show()

# Density vs Quality
plt.figure()
sns.boxplot(x='quality', y='density', data=df)
plt.title("Density vs Wine Quality")
plt.show()

# Fixed Acidity vs Quality
plt.figure()
sns.boxplot(x='quality', y='fixed acidity', data=df)
plt.title("Fixed Acidity vs Wine Quality")
plt.show()
df['quality_label'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

X = df.drop(['quality', 'quality_label', 'Id'], axis=1)
y = df['quality_label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
sgd = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd.fit(X_train, y_train)
sgd_pred = sgd.predict(X_test)

print("\nSGD Classifier Accuracy:", accuracy_score(y_test, sgd_pred))
print(classification_report(y_test, sgd_pred))
svc = SVC(kernel='rbf', random_state=42)
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)

print("\nSVC Accuracy:", accuracy_score(y_test, svc_pred))
print(classification_report(y_test, svc_pred))
models = {
    "Random Forest": rf_pred,
    "SGD Classifier": sgd_pred,
    "SVC": svc_pred
}

for model_name, predictions in models.items():
    plt.figure()
    sns.heatmap(confusion_matrix(y_test, predictions),
                annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()