import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os

# Heuristic matching to create weak labels for the TF-IDF baseline
# Since we don't have human labels yet, we bootstrap some intents based on AppleSupport patterns.
INTENT_RULES = {
    "Update Issue": ["update", "ios", "15", "16", "17", "froze", "stuck"],
    "Battery/Power": ["battery", "drain", "charge", "power", "dead"],
    "Account/ID": ["apple id", "password", "locked", "verification", "icloud"],
    "Hardware Damage": ["screen", "broke", "crack", "water", "dropped"],
    "App Store": ["purchase", "refund", "subscription", "app store", "download"]
}

def bootstrap_labels(text):
    text = str(text).lower()
    for intent, keywords in INTENT_RULES.items():
        if any(kw in text for kw in keywords):
            return intent
    return "General Inquiry"

def train_baseline_classifier(data_path="apple_conversations.csv"):
    if not os.path.exists(data_path):
        print(f"{data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    
    # 1. Apply Weak Supervision for Taxonomy
    df['intent'] = df['text_customer'].apply(bootstrap_labels)
    
    print("Class Distribution:")
    print(df['intent'].value_counts())
    
    # 2. Split (Conversation-level preservation applies naturally here as rows are pairs)
    X = df['text_customer']
    y = df['intent']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    
    # 3. Simple Baseline: TF-IDF + Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print("\n--- Baseline Results ---")
    print(classification_report(y_test, y_pred))
    
    # 4. Save Model
    os.makedirs("models", exist_ok=True)
    with open("models/intent_baseline.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    
    # 5. Create Golden Test Set (Unseen during training)
    golden = pd.DataFrame({'text_customer': X_test, 'true_intent': y_test, 'text_brand': df.loc[X_test.index, 'text_brand']})
    golden.to_csv("evaluation/golden_test.csv", index=False)
    
    return pipeline

if __name__ == "__main__":
    train_baseline_classifier()
