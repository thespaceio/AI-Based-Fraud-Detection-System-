import random
import uuid
from datetime import datetime, timedelta
import psycopg2
from faker import Faker
import numpy as np

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

# Nigerian mobile network prefixes
NG_PREFIXES = ['0803', '0806', '0805', '0807', '0810', '0813', '0814', '0816', '0903', '0906']

def nigerian_phone():
    return random.choice(NG_PREFIXES) + ''.join([str(random.randint(0,9)) for _ in range(7)])

def generate_users(n=1000):
    users = []
    for _ in range(n):
        user_id = uuid.uuid4()
        reg_date = fake.date_time_between(start_date='-2y', end_date='now')
        users.append({
            'user_id': user_id,
            'phone': nigerian_phone(),
            'email': fake.email(),
            'reg_date': reg_date,
            'avg_amount': np.random.gamma(shape=2, scale=500),  # typical ~1000 NGN
            'avg_freq': np.random.exponential(scale=2),  # ~2 tx/day
            'typical_lat': 6.5244 + np.random.normal(0, 0.1),  # Lagos region
            'typical_lon': 3.3792 + np.random.normal(0, 0.1),
            'typical_hour': np.random.choice(range(8, 21), p=[0.02]*8 + [0.12]*13)  # 8am-9pm
        })
    return users

def generate_transactions(users, days_back=90, fraud_ratio=0.005):
    transactions = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    for user in users:
        user_id = user['user_id']
        base_amt = user['avg_amount']
        base_freq = user['avg_freq']
        typical_hour = user['typical_hour']
        typical_loc = (user['typical_lat'], user['typical_lon'])
        
        # Number of transactions for this user (Poisson)
        n_tx = np.random.poisson(base_freq * days_back)
        
        for _ in range(n_tx):
            txn_time = fake.date_time_between(start_date=start_date, end_date=end_date)
            hour = txn_time.hour
            
            # Normal behaviour: close to typical hour and location
            if random.random() > 0.02:  # 98% normal
                is_fraud = False
                amount = max(10, np.random.gamma(shape=2, scale=base_amt/2))
                # location within 5km typical (using approximate)
                lat = typical_loc[0] + np.random.normal(0, 0.01)
                lon = typical_loc[1] + np.random.normal(0, 0.01)
                channel = np.random.choice(['mobile_app', 'ussd', 'web', 'pos'], p=[0.6, 0.3, 0.05, 0.05])
                device = f"device_{random.randint(1,1000)}"
            else:
                # Fraud transaction
                is_fraud = True
                fraud_type = np.random.choice(['amount_spike', 'location_jump', 'time_anomaly', 'device_change'], p=[0.4,0.3,0.2,0.1])
                if fraud_type == 'amount_spike':
                    amount = base_amt * np.random.uniform(5, 15)
                else:
                    amount = max(10, np.random.gamma(shape=2, scale=base_amt/2))
                
                if fraud_type == 'location_jump':
                    # Jump to another city (Abuja, PH, Kano)
                    jump_loc = {'lat': 9.0765, 'lon': 7.3986}  # Abuja
                    lat, lon = jump_loc['lat'] + np.random.normal(0, 0.05), jump_loc['lon'] + np.random.normal(0, 0.05)
                else:
                    lat, lon = typical_loc[0] + np.random.normal(0, 0.01), typical_loc[1] + np.random.normal(0, 0.01)
                
                if fraud_type == 'time_anomaly':
                    hour = np.random.choice([0,1,2,3,4,5,22,23])  # odd hours
                    txn_time = txn_time.replace(hour=hour)
                
                if fraud_type == 'device_change':
                    device = f"fraud_device_{random.randint(10000,99999)}"
                else:
                    device = f"device_{random.randint(1,1000)}"
                
                channel = np.random.choice(['mobile_app', 'ussd', 'web', 'pos'], p=[0.4,0.4,0.1,0.1])
            
            transactions.append({
                'transaction_id': uuid.uuid4(),
                'user_id': user_id,
                'amount': round(amount, 2),
                'transaction_time': txn_time,
                'location_lat': lat,
                'location_lon': lon,
                'device_id': device,
                'channel': channel,
                'ip_address': fake.ipv4(),
                'recipient_account': fake.iban()[:20],
                'is_fraud': is_fraud
            })
    return transactions

if __name__ == '__main__':
    print("Generating synthetic users...")
    users = generate_users(5000)
    print("Generating transactions...")
    txns = generate_transactions(users, days_back=90, fraud_ratio=0.005)
    fraud_count = sum(1 for t in txns if t['is_fraud'])
    print(f"Generated {len(txns)} transactions, {fraud_count} fraud ({fraud_count/len(txns)*100:.2f}%)")
    # Save to CSV or DB
    import pandas as pd
    pd.DataFrame(txns).to_csv('synthetic_transactions.csv', index=False)
    print("Saved to synthetic_transactions.csv")
