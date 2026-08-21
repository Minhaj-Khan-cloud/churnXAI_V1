import sqlite3
import pandas as pd
import os
from fpdf import FPDF

def init_db():
    db_path = "database/churn_actions.db"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS high_risk_customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT,
            risk_score REAL,
            top_reason TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_high_risk_customer(customer_name, risk_score, top_reason):
    db_path = "database/churn_actions.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO high_risk_customers (customer_name, risk_score, top_reason)
        VALUES (?, ?, ?)
    ''', (customer_name, risk_score, top_reason))
    conn.commit()
    conn.close()

def clean_text(text):
    if text is None:
        return ""
    return str(text).replace("—", "-").replace("–", "-").replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

def generate_retention_offer(customer_row):
    contract = customer_row.get('Contract', ['Month-to-month']).values[0] if hasattr(customer_row.get('Contract'), 'values') else 'Month-to-month'
    monthly_charges = customer_row.get('MonthlyCharges', [50]).values[0] if hasattr(customer_row.get('MonthlyCharges'), 'values') else 50
    payment_method = customer_row.get('PaymentMethod', ['Electronic check']).values[0] if hasattr(customer_row.get('PaymentMethod'), 'values') else 'Electronic check'
    
    offers = []
    
    if 'Month-to-month' in str(contract):
        offers.append("Contract Upgrade Offer: Switch to a 1-Year Annual Contract and get an instant 15% discount on monthly bills.")
    else:
        offers.append("Loyalty Reward: Thank you for your long-term commitment. Enjoy a free month on your next renewal cycle.")
        
    if float(monthly_charges) > 70:
        offers.append("High-Value Customer Perk: Complimentary 3-month access to Premium Streaming & Online Security add-ons.")
    else:
        offers.append("Service Enhancement: Free upgrade to high-speed fiber internet for the next billing cycle.")
        
    if 'Electronic check' in str(payment_method):
        offers.append("Payment Incentive: Switch to Auto-Pay (Bank Transfer/Credit Card) to receive a $10 cashback on your next bill.")
        
    return [clean_text(o) for o in offers]

def create_pdf_report(customer_id, raw_cust, risk_score, offers):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("helvetica", "B", 18)
    pdf.cell(0, 10, clean_text("ChurnXAI - Customer Retention Report"), 0, 1, "C")
    pdf.set_font("helvetica", "I", 10)
    pdf.cell(0, 6, clean_text("AI-Powered Customer Churn Prediction & Retention System"), 0, 1, "C")
    pdf.ln(10)
    
    pdf.set_font("helvetica", "B", 13)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, clean_text(f" Customer Details (ID / Index: {customer_id})"), 0, 1, "L", True)
    pdf.ln(3)
    
    pdf.set_font("helvetica", "", 11)
    for col in raw_cust.columns:
        val = raw_cust[col].values[0]
        pdf.cell(60, 7, clean_text(f"{col}:"), 0, 0)
        pdf.cell(0, 7, clean_text(f"{val}"), 0, 1)
        
    pdf.ln(5)
    
    pdf.set_font("helvetica", "B", 13)
    pdf.cell(0, 8, clean_text(f" Churn Risk Assessment: {risk_score}% (High Risk)"), 0, 1, "L", True)
    pdf.ln(5)
    
    pdf.set_font("helvetica", "B", 13)
    pdf.cell(0, 8, clean_text(" Recommended Retention Strategies:"), 0, 1, "L", True)
    pdf.ln(3)
    
    pdf.set_font("helvetica", "", 11)
    for i, offer in enumerate(offers, 1):
        pdf.multi_cell(0, 7, clean_text(f"{i}. {offer}"))
        pdf.ln(2)
        
    pdf.ln(10)
    pdf.set_font("helvetica", "I", 9)
    pdf.cell(0, 10, clean_text("Generated locally via ChurnXAI Rule-Based Engine - 100% Offline & Secure"), 0, 1, "C")
    
    # স্ট্রিমলিটের বাইনারি এরর চিরতরে দূর করার জন্য স্ট্যান্ডার্ড bytes এ রূপান্তর
    output_data = pdf.output()
    if isinstance(output_data, bytearray):
        return bytes(output_data)
    elif isinstance(output_data, str):
        return output_data.encode("latin-1")
    return bytes(output_data)