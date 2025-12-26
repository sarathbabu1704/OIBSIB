import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
df = pd.read_csv("Popular_Spotify_Songs.csv", encoding="latin1")
print("\n--- First 5 Rows ---")
print(df.head())
print("\n--- Missing Values ---")
print(df.isnull().sum())
df = df.dropna()   
df["streams"] = pd.to_numeric(df["streams"], errors="coerce")
df["in_spotify_playlists"] = pd.to_numeric(df["in_spotify_playlists"], errors="coerce")
df["in_spotify_charts"] = pd.to_numeric(df["in_spotify_charts"], errors="coerce")
df = df.dropna(subset=["streams"])
print("\n--- Descriptive Statistics ---")
print(df.describe())
print("\nAverage Streams: ", df["streams"].mean())
print("\n--- Descriptive Statistics ---")
print(df.describe())
print("\nAverage Streams: ", df["streams"].mean())
print("Average Danceability: ", df["danceability_%"].mean())
features = [
    "streams",
    "in_spotify_playlists",
    "in_spotify_charts",
    "danceability_%",
    "energy_%",
    "valence_%",
    "acousticness_%"
]
X = df[features]
# Standardize numerical values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
wcss = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(range(2, 10), wcss, marker="o")
plt.title("Elbow Method (Choose Optimal K)")
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.show()
k = 4
model = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = model.fit_predict(X_scaled)

print("\n--- Sample Cluster Assignments ---")
print(df[["track_name", "artist(s)_name", "Cluster"]].head())
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=df["streams"],
    y=df["danceability_%"],
    hue=df["Cluster"],
    palette="viridis",
    s=80
)
plt.title("Spotify Track Segmentation")
plt.xlabel("Streams")
plt.ylabel("Danceability %")
plt.show()

#Bar Chart
cluster_avg = df.groupby("Cluster")["streams"].mean()

plt.figure(figsize=(8, 6))
cluster_avg.plot(kind="bar")
plt.title("Average Streams per Cluster (Bar Chart)")
plt.xlabel("Cluster")
plt.ylabel("Average Streams")
plt.show()
cluster_summary = df.groupby("Cluster")[features].mean()
print("\n--- Cluster Summary ---")
print(cluster_summary)

print("\n--- Insights ---")
for c, row in cluster_summary.iterrows():
    print(f"\nCluster {c}:")
    print(f"- Avg Streams: {row['streams']:.2f}")
    print(f"- Danceability: {row['danceability_%']:.2f}")
    print(f"- Energy: {row['energy_%']:.2f}")

    # Interpret cluster behavior
    if row["streams"] > df["streams"].mean():
        print("  → Popular / Viral songs")
    else:
        print("  → Less popular tracks")

    if row["danceability_%"] > 60:
        print("  → Highly danceable music")
    elif row["energy_%"] > 70:
        print("  → High energy tracks")