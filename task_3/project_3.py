import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
df = pd.read_csv("AB_NYC_2019.csv")

print("Original Dataset Shape:", df.shape)
print("\nDataset Preview:")
print(df.head())
df = df[['name', 'neighbourhood_group', 'room_type', 'price']]
df.dropna(subset=['name'], inplace=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]
    
    return " ".join(words)
df['cleaned_name'] = df['name'].apply(clean_text)

print("\nCleaned Text Sample:")
print(df[['name', 'cleaned_name']].head())
df = df[df['cleaned_name'].str.strip() != ""]
print("\nDataset Shape After Cleaning:", df.shape)
df.to_csv("AB_NYC_2019_CLEANED.csv", index=False)
print("\n✅ Data cleaning completed successfully!")
print("📁 Output file saved as: AB_NYC_2019_CLEANED.csv")