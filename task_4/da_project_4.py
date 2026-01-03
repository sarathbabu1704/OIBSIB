import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df = pd.read_csv("Twitter_Data.csv")
df.dropna(inplace=True)
print("Dataset Shape:", df.shape)
print(df.head())
X = df['clean_text']
y = df['category']
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_tfidf = tfidf.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)
model = LinearSVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
sentiment_map = {
    -1: "Negative",
     0: "Neutral",
     1: "Positive"
}

df['sentiment'] = df['category'].map(sentiment_map)
sentiment_counts = df['sentiment'].value_counts()
# Bar Chart
plt.figure(figsize=(6,4))
sentiment_counts.plot(kind='bar')
plt.title("Sentiment Distribution of Twitter Data")
plt.xlabel("Sentiment")
plt.ylabel("Tweet Count")
plt.tight_layout()
plt.show()
# Pie Chart
plt.figure(figsize=(5,5))
sentiment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title("Sentiment Distribution (%)")
plt.ylabel("")
plt.tight_layout()
plt.show()
cm = confusion_matrix(y_test, y_pred)

labels = ["Negative", "Neutral", "Positive"]

plt.figure(figsize=(6,5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks([0,1,2], labels)
plt.yticks([0,1,2], labels)

# Add values inside matrix
for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
df['tweet_length'] = df['clean_text'].apply(len)
plt.figure(figsize=(6,4))
df.boxplot(column='tweet_length', by='sentiment')

df.to_csv("Twitter_Sentiment_Analysis_Output.csv", index=False)
print("\n✅ Output saved as Twitter_Sentiment_Analysis_Output.csv")
