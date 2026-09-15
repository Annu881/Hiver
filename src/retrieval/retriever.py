import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class HistoricalRetriever:
    def __init__(self, data_path="apple_conversations.csv", build=False, model_path="models/retriever.pkl"):
        if build:
            self.df = pd.read_csv(data_path).dropna(subset=['text_customer']).reset_index(drop=True)
            self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            self.vectors = self.vectorizer.fit_transform(self.df['text_customer'])
            print(f"Retriever initialized with {len(self.df)} conversations.")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump((self.vectorizer, self.vectors, self.df), f)
        else:
            with open(model_path, "rb") as f:
                self.vectorizer, self.vectors, self.df = pickle.load(f)

    def retrieve(self, query, top_k=3):
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.vectors).flatten()
        top_indices = sims.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "customer_msg": self.df.iloc[idx]['text_customer'],
                "brand_reply": self.df.iloc[idx]['text_brand'],
                "score": float(sims[idx])
            })
        return results

if __name__ == "__main__":
    retriever = HistoricalRetriever(build=True)
