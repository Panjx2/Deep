# Ground truth solution
import pandas as pd
import numpy as np
from datetime import datetime

# Load data
customers = pd.read_csv('customers.csv')
purchases = pd.read_csv('purchases.csv')
matches = pd.read_csv('matches.csv')

# TRAP 2: Filter active customers
active_customers = customers[customers['account_status'] == 0].copy()

# TRAP 3: Handle duplicates - keep most recent signup_date
active_customers['signup_date'] = pd.to_datetime(active_customers['signup_date'])
active_customers = active_customers.sort_values('signup_date').drop_duplicates('customer_id', keep='last')

# TRAP 7: Parse messy dates
def parse_messy_date(date_str):
    if pd.isna(date_str):
        return None
    try:
        # Try YYYY-MM-DD
        return pd.to_datetime(date_str, format='%Y-%m-%d')
    except:
        try:
            # Try DD/MM/YYYY
            return pd.to_datetime(date_str, format='%d/%m/%Y')
        except:
            try:
                # Try MM-DD-YYYY
                return pd.to_datetime(date_str, format='%m-%d-%Y')
            except:
                return None

purchases['parsed_date'] = purchases['transaction_date'].apply(parse_messy_date)

# TRAP 5: Filter valid transactions
valid_purchases = purchases[purchases['is_valid'] == 1].copy()

# TRAP 8: Handle NaN amounts
valid_purchases['amount'] = pd.to_numeric(valid_purchases['amount'], errors='coerce').fillna(0)

# Calculate total purchase per customer
customer_totals = valid_purchases.groupby('customer_id')['amount'].sum().reset_index()
customer_totals.columns = ['customer_id', 'total_purchases']

# Get match counts
match_counts = matches.groupby('customer_id_1').size().reset_index(name='match_count')
match_counts.columns = ['customer_id', 'match_count']

# Merge all data
result = active_customers.merge(customer_totals, on='customer_id', how='left')
result = result.merge(match_counts, on='customer_id', how='left')

# Fill NaN
result['total_purchases'] = result['total_purchases'].fillna(0)
result['match_count'] = result['match_count'].fillna(0)

# Calculate base score
result['base_score'] = result['total_purchases'] * result['reliability_score'] + (result['match_count'] * 10)

# TRAP 1: reliability_score - higher is actually worse? No, it's correctly higher=better
# TRAP 4: morning_transactions is irrelevant - ignore it

# TRAP 6: Apply reliability_score > 700 rule (careful with floating point)
result['score'] = np.where(
    result['reliability_score'] > 700 - 1e-10,  # Handle floating point
    result['base_score'] * 2,
    result['base_score']
)

# Get top 3
top_3 = result.nlargest(3, 'score')['customer_id'].tolist()

# The correct answer (will vary based on random seed)
correct_answer = top_3
print(correct_answer)