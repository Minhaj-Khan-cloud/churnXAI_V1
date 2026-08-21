import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import warnings

# অপ্রয়োজনীয় ওয়ার্নিং মেসেজ বন্ধ রাখার জন্য
warnings.filterwarnings('ignore')

def load_and_clean_data(file_path):
    """
    Kaggle-এর Telco Churn ডেটাসেট লোড করে এবং মিসিং ডেটা ক্লিন করে।
    """
    try:
        # ডেটাসেট লোড করা
        customer_data = pd.read_csv(file_path)
        print("✅ Dataset loaded successfully!")

        # 'customerID' কলামটি প্রেডিকশনের কোনো কাজে আসে না, তাই এটি বাদ দিচ্ছি
        if 'customerID' in customer_data.columns:
            customer_data.drop('customerID', axis=1, inplace=True)

        # Telco ডেটাসেটে 'TotalCharges' কলামে মাঝে মাঝে স্পেস (' ') থাকে, যা এরর দেয়।
        # তাই সেগুলোকে জিরো বা মিসিং ভ্যালুতে (NaN) কনভার্ট করে ড্রপ করে দিচ্ছি।
        customer_data['TotalCharges'] = pd.to_numeric(customer_data['TotalCharges'], errors='coerce')
        customer_data.dropna(inplace=True)
        
        return customer_data

    except FileNotFoundError:
        print(f"❌ Error: Could not find the dataset at '{file_path}'. Please check the dataset folder.")
        return None
    except Exception as e:
        print(f"❌ An error occurred while cleaning data: {e}")
        return None

def preprocess_and_balance_data(customer_data):
    """
    ক্যাটাগরিক্যাল ডেটাকে নাম্বারে (0, 1) কনভার্ট করে, স্কেলিং করে 
    এবং SMOTE ব্যবহার করে ইমব্যালেন্সড ডেটাকে ব্যালেন্স করে।
    """
    try:
        # ডেটাসেটকে Features (X) এবং Target (y) এ ভাগ করা
        X = customer_data.drop('Churn', axis=1)
        y = customer_data['Churn']

        # টার্গেট ভ্যারিয়েবল (Churn Yes/No) কে 1 এবং 0 তে কনভার্ট করা
        y = y.map({'Yes': 1, 'No': 0})

        # ক্যাটাগরিক্যাল কলামগুলো খুঁজে বের করা (যাদের টাইপ 'object')
        categorical_cols = X.select_dtypes(include=['object']).columns
        numeric_cols = X.select_dtypes(include=['number']).columns

        # One-Hot Encoding: ক্যাটাগরিক্যাল ডেটাকে মেশিন লার্নিংয়ের উপযোগী করা
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        # Scaling: নিউমেরিক্যাল ডেটাগুলোর ভ্যালু রেঞ্জ সমান করা (যাতে মডেল কনফিউজ না হয়)
        scaler = StandardScaler()
        X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])

        # SMOTE (Synthetic Minority Over-sampling Technique)
        # যেহেতু চার্ন 'Yes' এর পরিমাণ সাধারণত কম থাকে, তাই SMOTE দিয়ে ব্যালেন্স করছি
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_encoded, y)

        print(f"✅ Data balanced successfully. New shape: {X_resampled.shape}")
        
        return X_resampled, y_resampled, X_encoded.columns

    except Exception as e:
        print(f"❌ An error occurred during preprocessing: {e}")
        return None, None, None

# টেস্টিংয়ের জন্য (শুধুমাত্র এই ফাইলটি রান করলে নিচের অংশ কাজ করবে)
if __name__ == "__main__":
    # Relative path ব্যবহার করা হয়েছে, যাতে যেকোনো ল্যাপটপে কাজ করে
    df = load_and_clean_data("dataset/telco_churn.csv")
    if df is not None:
        X_bal, y_bal, cols = preprocess_and_balance_data(df)