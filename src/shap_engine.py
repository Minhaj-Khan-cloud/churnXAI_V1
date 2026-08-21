import shap
import matplotlib.pyplot as plt
import joblib
import warnings

# অপ্রয়োজনীয় ওয়ার্নিং বন্ধ রাখার জন্য
warnings.filterwarnings('ignore')

def load_explainer(model_path, X_background):
    """
    Random Forest মডেলের জন্য SHAP TreeExplainer তৈরি করবে।
    """
    try:
        model = joblib.load(model_path)
        # Random Forest ক্লাসিফায়ারের জন্য TreeExplainer সরাসরি ও নিখুঁত কাজ করে
        explainer = shap.TreeExplainer(model)
        return explainer
    except Exception as e:
        print(f"❌ Explainer Load Error: {e}")
        return None

def generate_waterfall_plot(explainer, customer_data):
    """
    একজন নির্দিষ্ট কাস্টমারের জন্য SHAP Waterfall Plot জেনারেট করবে।
    """
    try:
        if explainer is None:
            raise ValueError("Explainer object is None")
            
        # SHAP ভ্যালু হিসাব করা
        shap_values = explainer(customer_data)
        
        # Random Forest Churn (Class 1) এর জন্য সঠিক শেপ ইনডেক্স করা
        if len(shap_values.shape) == 3:
            shap_obj = shap_values[0, :, 1]
        else:
            shap_obj = shap_values[0]

        # ওয়াটারফল প্লট তৈরি করা
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(shap_obj, show=False)
        plt.tight_layout()
        
        return fig
    except Exception as e:
        print(f"❌ Error generating waterfall plot: {e}")
        return None