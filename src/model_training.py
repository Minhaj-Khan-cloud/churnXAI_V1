import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings

# অপ্রয়োজনীয় ওয়ার্নিং মেসেজ বন্ধ রাখার জন্য
warnings.filterwarnings('ignore')

def train_and_evaluate_models(X, y):
    """
    Random Forest এবং Logistic Regression মডেল ট্রেইন করে 
    এবং পারফর্মেন্স তুলনা করে সেরা মডেলটি সেভ করে।
    """
    try:
        print("⚙️ Splitting data into Training and Testing sets...")
        # ডেটা ভাগ করা (৮০% ট্রেনিং, ২০% টেস্টিং)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # ১. Logistic Regression (বেসলাইন মডেল)
        print("🧠 Training Logistic Regression...")
        log_reg = LogisticRegression(max_iter=1000, random_state=42)
        log_reg.fit(X_train, y_train)
        
        # সম্ভাব্যতা (Probability) বের করছি ROC-AUC স্কোরের জন্য
        log_prob = log_reg.predict_proba(X_test)[:, 1]
        log_roc = roc_auc_score(y_test, log_prob)

        # ২. Random Forest (অ্যাডভান্সড ট্রি মডেল)
        print("🌲 Training Random Forest Classifier...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        rf_prob = rf_model.predict_proba(X_test)[:, 1]
        rf_roc = roc_auc_score(y_test, rf_prob)

        # পারফর্মেন্স তুলনা
        print("\n📊 --- Model Performance ---")
        print(f"Logistic Regression ROC-AUC: {log_roc:.4f}")
        print(f"Random Forest ROC-AUC: {rf_roc:.4f}")

        # সেরা মডেল সিলেক্ট করা (আমাদের ক্ষেত্রে যার ROC-AUC বেশি)
        best_model = rf_model if rf_roc > log_roc else log_reg
        best_model_name = "Random Forest" if rf_roc > log_roc else "Logistic Regression"
        print(f"\n🏆 Best Model Selected: {best_model_name}")

        # মডেল সেভ করা (যাতে ওয়েবসাইট বানানোর সময় বারবার ট্রেইন করতে না হয়)
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        model_path = os.path.join(models_dir, "best_churn_model.pkl")
        joblib.dump(best_model, model_path)
        print(f"✅ Model saved successfully at: {model_path}")

        # SHAP এক্সপ্লেনারের জন্য X_train রিটার্ন করা হচ্ছে
        return best_model, X_train

    except Exception as e:
        print(f"❌ Error during model training: {e}")
        return None, None

# টেস্টিংয়ের জন্য 
if __name__ == "__main__":
    from data_prep import load_and_clean_data, preprocess_and_balance_data
    
    # ডেটা লোড এবং প্রসেস করে এই মডেলে পাস করা
    df = load_and_clean_data("dataset/telco_churn.csv")
    if df is not None:
        X_bal, y_bal, cols = preprocess_and_balance_data(df)
        if X_bal is not None:
            train_and_evaluate_models(X_bal, y_bal)