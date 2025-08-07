import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import skfuzzy as fuzz
import numpy as np

# Step 1: Load data
df = pd.read_csv(r"C:\Users\souma\OneDrive\Documents\Custom Office Templates\Desktop\Symptom2Disease.csv")
print(df.columns)
print(df.head())
print(df.isnull().sum())

# Step 2: Convert text to features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
X = vectorizer.fit_transform(df['text']).toarray()
print("TF-IDF feature shape:", X.shape)

# Step 3: Fuzzy C-Means Clustering
X_T = X.T
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    data=X_T, c=4, m=2, error=0.005, maxiter=1000
)
labels = np.argmax(u, axis=0)
df['Cluster'] = labels
print(df[['label', 'Cluster']].head())

# Step 4: Save result
df.to_csv("clustered_symptom_output.csv", index=False)
print("Output saved to clustered_symptom_output.csv")

import os
print("CSV saved at:", os.path.abspath("clustered_symptom_output.csv"))
