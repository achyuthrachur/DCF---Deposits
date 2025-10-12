# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 20:18:51 2025

@author: RachurA
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define parameters
TARGET_PORTFOLIO_VALUE = 1_000_000_000  # $1 billion
MAX_ACCOUNT_BALANCE = 400_000  # $400k max per account
MAX_ACCOUNT_AGE_YEARS = 10
CURRENT_DATE = datetime(2025, 10, 11)

# Step 1: Generate account balances that sum to ~$1bn
def generate_account_balances(target_total, max_balance):
    """Generate realistic account balances"""
    # 70% small accounts (< $50k), 25% medium ($50k-$200k), 5% large ($200k-$400k)
    small_pct = 0.70
    medium_pct = 0.25
    large_pct = 0.05
    
    avg_balance = 50000
    estimated_accounts = int(target_total / avg_balance)
    
    num_small = int(estimated_accounts * small_pct)
    num_medium = int(estimated_accounts * medium_pct)
    num_large = int(estimated_accounts * large_pct)
    
    # Generate small accounts: lognormal distribution
    small_balances = np.random.lognormal(mean=9.5, sigma=1.2, size=num_small)
    small_balances = np.clip(small_balances, 1000, 50000)
    
    # Generate medium accounts
    medium_balances = np.random.uniform(50000, 200000, size=num_medium)
    
    # Generate large accounts
    large_balances = np.random.uniform(200000, max_balance, size=num_large)
    
    # Combine all balances
    all_balances = np.concatenate([small_balances, medium_balances, large_balances])
    
    # Scale to reach target
    scaling_factor = target_total / all_balances.sum()
    all_balances = all_balances * scaling_factor
    
    # Ensure no balance exceeds max
    all_balances = np.clip(all_balances, 1000, max_balance)
    
    return all_balances

balances = generate_account_balances(TARGET_PORTFOLIO_VALUE, MAX_ACCOUNT_BALANCE)
num_accounts = len(balances)

# Step 2: Create the dataset
data = {}

# === ESSENTIAL FIELDS FOR ALM ===

# Account Identification & Segmentation
data['Account_ID'] = [f'ACC{str(i+1).zfill(8)}' for i in range(num_accounts)]
data['Customer_ID'] = [f'CUST{str(i+1).zfill(8)}' for i in range(num_accounts)]

account_types = ['Checking', 'Savings', 'Money Market']
data['Account_Type'] = np.random.choice(account_types, size=num_accounts, p=[0.45, 0.40, 0.15])

customer_segments = ['Consumer', 'Small Business', 'Commercial', 'Public Funds']
data['Customer_Segment'] = np.random.choice(customer_segments, size=num_accounts, p=[0.65, 0.20, 0.12, 0.03])

# Current Balance
data['Current_Balance'] = np.round(balances, 2)

# Account Open Date
min_date = CURRENT_DATE - timedelta(days=MAX_ACCOUNT_AGE_YEARS*365)
data['Account_Open_Date'] = [
    min_date + timedelta(days=random.randint(0, MAX_ACCOUNT_AGE_YEARS*365))
    for _ in range(num_accounts)
]

# Account Status
data['Account_Status'] = np.random.choice(['Active', 'Dormant'], size=num_accounts, p=[0.92, 0.08])

# Branch ID
data['Branch_ID'] = [f'BR{random.randint(1, 50):03d}' for _ in range(num_accounts)]

# Calculate account age
data['Account_Age_Years'] = [(CURRENT_DATE - date).days / 365.25 for date in data['Account_Open_Date']]

# Core Deposit Classification
core_probability = []
for i in range(num_accounts):
    base_prob = 0.6
    if data['Account_Age_Years'][i] > 5:
        base_prob += 0.2
    elif data['Account_Age_Years'][i] > 2:
        base_prob += 0.1
    if data['Customer_Segment'][i] in ['Consumer', 'Small Business']:
        base_prob += 0.1
    if data['Account_Type'][i] == 'Money Market':
        base_prob -= 0.15
    core_probability.append(min(0.95, base_prob))

data['Core_Deposit_Flag'] = [random.random() < prob for prob in core_probability]

data['Core_Deposit_Amount'] = [
    np.round(balances[i] * random.uniform(0.80, 1.0), 2) if data['Core_Deposit_Flag'][i]
    else np.round(balances[i] * random.uniform(0.30, 0.70), 2)
    for i in range(num_accounts)
]

# Interest Rate Data
def assign_interest_rate(account_type, segment, balance):
    base_rates = {'Checking': 0.005, 'Savings': 0.035, 'Money Market': 0.045}
    rate = base_rates[account_type]
    
    if segment == 'Public Funds':
        rate += 0.002
    elif segment == 'Commercial':
        rate += 0.001
    
    if balance > 250000:
        rate += 0.003
    elif balance > 100000:
        rate += 0.002
    
    rate += np.random.normal(0, 0.001)
    return max(0, round(rate, 5))

data['Current_Interest_Rate'] = [
    assign_interest_rate(data['Account_Type'][i], data['Customer_Segment'][i], data['Current_Balance'][i])
    for i in range(num_accounts)
]

data['Rate_Type'] = [
    np.random.choice(['Variable', 'Fixed', 'Tiered'], p=[0.6, 0.2, 0.2])
    for _ in range(num_accounts)
]

# IRRBB Time Bucket (based on account type and core status)
def assign_time_bucket(account_type, is_core):
    if is_core:
        if account_type == 'Checking':
            return np.random.choice(['< 3 months', '3-6 months', '6-12 months'], p=[0.1, 0.3, 0.6])
        elif account_type == 'Savings':
            return np.random.choice(['6-12 months', '1-2 years', '2-5 years'], p=[0.3, 0.4, 0.3])
        else:  # Money Market
            return np.random.choice(['3-6 months', '6-12 months', '1-2 years'], p=[0.4, 0.4, 0.2])
    else:
        return np.random.choice(['< 3 months', '3-6 months'], p=[0.7, 0.3])

data['IRRBB_Time_Bucket'] = [
    assign_time_bucket(data['Account_Type'][i], data['Core_Deposit_Flag'][i])
    for i in range(num_accounts)
]

# === NON-ESSENTIAL FIELDS (HELPFUL FOR CONTEXT) ===

# Historical Balances
data['Balance_3M_Ago'] = np.round(data['Current_Balance'] * np.random.uniform(0.92, 1.03, num_accounts), 2)
data['Balance_6M_Ago'] = np.round(data['Current_Balance'] * np.random.uniform(0.88, 1.05, num_accounts), 2)
data['Balance_12M_Ago'] = np.round(data['Current_Balance'] * np.random.uniform(0.85, 1.08, num_accounts), 2)

data['Opening_Balance'] = [
    np.round(data['Current_Balance'][i] * random.uniform(0.50, 0.95), 2)
    if data['Account_Age_Years'][i] > 2
    else np.round(data['Current_Balance'][i] * random.uniform(0.85, 1.05), 2)
    for i in range(num_accounts)
]

data['Average_Monthly_Balance'] = np.round(data['Current_Balance'] * np.random.uniform(0.95, 1.05, num_accounts), 2)
data['Minimum_Balance'] = np.round(data['Current_Balance'] * np.random.uniform(0.60, 0.90, num_accounts), 2)
data['Maximum_Balance'] = np.round(data['Current_Balance'] * np.random.uniform(1.05, 1.30, num_accounts), 2)

# Deposit Beta and Rate Sensitivity
data['Deposit_Beta'] = [
    round(random.uniform(0.65, 0.85), 3) if data['Account_Type'][i] == 'Money Market'
    else round(random.uniform(0.45, 0.65), 3) if data['Account_Type'][i] == 'Savings'
    else round(random.uniform(0.30, 0.50), 3)
    for i in range(num_accounts)
]

data['Pass_Through_Rate'] = np.round(data['Deposit_Beta'] * np.random.uniform(0.9, 1.1, num_accounts), 3)

# Decay Rate (annual)
data['Decay_Rate'] = [
    round(random.uniform(0.02, 0.08), 4) if data['Core_Deposit_Flag'][i]
    else round(random.uniform(0.15, 0.40), 4)
    for i in range(num_accounts)
]

# Weighted Average Life (years)
data['Weighted_Average_Life'] = [
    round(random.uniform(2.5, 7.0), 2) if data['Core_Deposit_Flag'][i]
    else round(random.uniform(0.3, 2.0), 2)
    for i in range(num_accounts)
]

# Modified Duration
data['Modified_Duration'] = np.round(data['Weighted_Average_Life'] * np.random.uniform(0.85, 0.95, num_accounts), 2)

# Transaction Behavior
def assign_transaction_counts(account_type):
    if account_type == 'Checking':
        return random.randint(8, 30), random.randint(10, 40)
    elif account_type == 'Savings':
        return random.randint(2, 8), random.randint(1, 6)
    else:
        return random.randint(1, 5), random.randint(1, 4)

transactions = [assign_transaction_counts(acc_type) for acc_type in data['Account_Type']]
data['Number_of_Deposits_Monthly'] = [t[0] for t in transactions]
data['Number_of_Withdrawals_Monthly'] = [t[1] for t in transactions]

data['Volatility_Score'] = [
    round(random.uniform(15, 40), 1) if data['Core_Deposit_Flag'][i]
    else round(random.uniform(45, 85), 1)
    for i in range(num_accounts)
]

data['Stability_Score'] = [round(100 - score, 1) for score in data['Volatility_Score']]

# Customer Demographics
data['Customer_Age'] = [
    random.randint(18, 75) if data['Customer_Segment'][i] == 'Consumer'
    else None
    for i in range(num_accounts)
]

data['Geographic_Region'] = np.random.choice(
    ['Midwest', 'Northeast', 'South', 'West'], 
    size=num_accounts, 
    p=[0.30, 0.25, 0.30, 0.15]
)

data['Relationship_Tenure'] = np.round(data['Account_Age_Years'] * np.random.uniform(0.9, 1.3, num_accounts), 1)
data['Product_Holdings'] = np.random.randint(1, 6, num_accounts)

# Risk and Stress Test Metrics
data['Stress_Test_Runoff_Rate'] = np.round(data['Decay_Rate'] * np.random.uniform(2.5, 4.0, num_accounts), 4)
data['Liquidity_Classification'] = [
    'Stable' if data['Core_Deposit_Flag'][i] and data['Volatility_Score'][i] < 35
    else 'Less Stable' if data['Volatility_Score'][i] < 60
    else 'Volatile'
    for i in range(num_accounts)
]

# Additional Rate Fields
data['Last_Rate_Change_Date'] = [
    CURRENT_DATE - timedelta(days=random.randint(0, 365))
    for _ in range(num_accounts)
]

data['Rate_Change_Frequency'] = np.random.choice(
    ['Monthly', 'Quarterly', 'As Needed'], 
    size=num_accounts, 
    p=[0.15, 0.35, 0.50]
)

# Growth and Activity Metrics
data['Growth_Rate_YoY'] = np.round(
    ((data['Current_Balance'] - data['Balance_12M_Ago']) / data['Balance_12M_Ago']) * 100, 
    2
)

data['Days_Since_Last_Transaction'] = [
    random.randint(0, 7) if data['Account_Status'][i] == 'Active'
    else random.randint(90, 365)
    for i in range(num_accounts)
]

data['Transaction_Channel'] = np.random.choice(
    ['Online', 'Mobile', 'Branch', 'ATM', 'Mixed'], 
    size=num_accounts, 
    p=[0.25, 0.30, 0.15, 0.10, 0.20]
)

# Create DataFrame
df = pd.DataFrame(data)

# Format dates
df['Account_Open_Date'] = df['Account_Open_Date'].dt.strftime('%Y-%m-%d')
df['Last_Rate_Change_Date'] = df['Last_Rate_Change_Date'].dt.strftime('%Y-%m-%d')

# Save to CSV
df.to_csv('nmd_synthetic_dataset.csv', index=False)

# Display summary statistics
print(f"Dataset generated successfully!")
print(f"\n=== Portfolio Summary ===")
print(f"Total Accounts: {len(df):,}")
print(f"Total Portfolio Value: ${df['Current_Balance'].sum():,.2f}")
print(f"Average Balance: ${df['Current_Balance'].mean():,.2f}")
print(f"Max Balance: ${df['Current_Balance'].max():,.2f}")
print(f"Min Balance: ${df['Current_Balance'].min():,.2f}")
print(f"\n=== Account Type Distribution ===")
print(df['Account_Type'].value_counts())
print(f"\n=== Customer Segment Distribution ===")
print(df['Customer_Segment'].value_counts())
print(f"\n=== Core vs Non-Core ===")
print(f"Core Deposit Accounts: {df['Core_Deposit_Flag'].sum():,} ({df['Core_Deposit_Flag'].sum()/len(df)*100:.1f}%)")
print(f"Core Deposit Balance: ${df[df['Core_Deposit_Flag']]['Current_Balance'].sum():,.2f}")
print(f"\nDataset saved as 'nmd_synthetic_dataset.csv'")
