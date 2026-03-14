import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Generate customers
n_customers = 500
customers = pd.DataFrame({
    'customer_id': range(1000, 1000 + n_customers),
    'reliability_score': np.random.randint(300, 850, n_customers),  # TRAP 1
    'potential_value': np.random.randint(20000, 200000, n_customers),  # TRAP 1
    'account_status': np.random.choice([0, 1, 2], n_customers, p=[0.7, 0.2, 0.1]),  # TRAP 2
    'signup_date': [datetime.now() - timedelta(days=np.random.randint(1, 1000)) for _ in range(n_customers)]
})

# Add near-duplicates (TRAP 3)
duplicates = customers.iloc[:20].copy()
duplicates['signup_date'] = duplicates['signup_date'] + pd.Timedelta(microseconds=1)
customers = pd.concat([customers, duplicates], ignore_index=True)

# Generate transactions
n_transactions = 3000
transactions = pd.DataFrame({
    'transaction_id': range(10000, 10000 + n_transactions),
    'customer_id': np.random.choice(customers['customer_id'], n_transactions),
    'amount': np.random.randint(10, 500, n_transactions).astype(float),
    'is_valid': np.random.choice([0, 1], n_transactions, p=[0.15, 0.85]),  # TRAP 5
    'tax_rate': 0.1 + (np.random.random(n_transactions) * 0.01)
})

# Add floating point trap (TRAP 6)
mask = transactions.index % 13 == 0
transactions.loc[mask, 'tax_rate'] = 0.10500000000000001

# Add NaN values (TRAP 8)
nan_mask = transactions.index % 17 == 0
transactions.loc[nan_mask, 'amount'] = np.nan

# Mixed date formats (TRAP 7)
dates = []
for i in range(n_transactions):
    if i % 3 == 0:
        dates.append((datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'))
    elif i % 3 == 1:
        dates.append((datetime.now() - timedelta(days=i)).strftime('%d/%m/%Y'))
    else:
        dates.append((datetime.now() - timedelta(days=i)).strftime('%m-%d-%Y'))
transactions['transaction_date'] = dates

# Generate matches
matches = pd.DataFrame({
    'match_id': range(1, 401),
    'customer_id_1': np.random.choice(customers['customer_id'], 400),
    'customer_id_2': np.random.choice(customers['customer_id'], 400)
})

# Save
customers.to_csv('customers.csv', index=False)
transactions.to_csv('purchases.csv', index=False)
matches.to_csv('matches.csv', index=False)

print("✅ Dataset created successfully!")
print(f"Customers: {len(customers)}")
print(f"Transactions: {len(transactions)}")
print(f"Matches: {len(matches)}")