import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
from datetime import datetime

# ========== 1. Load data ==========
print("Loading synthetic transactions...")
df = pd.read_csv('synthetic_transactions.csv')
df['transaction_time'] = pd.to_datetime(df['transaction_time'])

print(f"Total transactions: {len(df)}")
print(f"Fraud ratio: {df['is_fraud'].mean():.4f}")

# ========== 2. Feature Engineering (same as in API) ==========
def haversine(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Compute user-level aggregates (for baseline)
user_agg = df.groupby('user_id').agg({
    'amount': ['mean', 'std'],
    'location_lat': 'mean',
    'location_lon': 'mean',
    'transaction_time': lambda x: x.dt.hour.mean()  # typical hour
}).reset_index()
user_agg.columns = ['user_id', 'avg_amount', 'std_amount', 'typical_lat', 'typical_lon', 'typical_hour']

# Merge back to each transaction
df = df.merge(user_agg, on='user_id', how='left')
df['std_amount'] = df['std_amount'].fillna(df['avg_amount'] * 0.5)  # if only 1 txn

# Amount features
df['amount_z_score'] = (df['amount'] - df['avg_amount']) / df['std_amount'].clip(lower=0.01)
df['amount_ratio'] = df['amount'] / df['avg_amount'].clip(lower=0.01)

# Time features
df['hour'] = df['transaction_time'].dt.hour
df['day_of_week'] = df['transaction_time'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_night'] = ((df['hour'] < 5) | (df['hour'] > 22)).astype(int)
df['hour_deviation'] = abs(df['hour'] - df['typical_hour'])

# Location features
df['distance_km'] = haversine(
    df['location_lat'].values, df['location_lon'].values,
    df['typical_lat'].values, df['typical_lon'].values
)
df['location_anomaly'] = (df['distance_km'] > 50).astype(int)

# Channel encoding
le_channel = LabelEncoder()
df['channel_code'] = le_channel.fit_transform(df['channel'])

# Select final features for ML
feature_cols = [
    'amount', 'amount_z_score', 'amount_ratio',
    'hour', 'is_night', 'hour_deviation',
    'distance_km', 'location_anomaly',
    'channel_code', 'is_weekend'
]
X = df[feature_cols].fillna(0)
y = df['is_fraud'].astype(int)

print(f"Feature matrix shape: {X.shape}")

# ========== 3. Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (important for Isolation Forest)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 4. Train Isolation Forest (Unsupervised) ==========
print("\nTraining Isolation Forest...")
iso_forest = IsolationForest(
    contamination=df['is_fraud'].mean(),  # expected fraud ratio
    random_state=42,
    n_estimators=100
)
iso_forest.fit(X_train_scaled)
# Anomaly score: -1 for anomaly, 1 for normal
y_pred_iso = iso_forest.predict(X_test_scaled)
iso_anomaly_score = -iso_forest.score_samples(X_test_scaled)  # higher = more anomalous

# ========== 5. Train Random Forest (Supervised) ==========
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',  # important for imbalanced fraud
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
y_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]  # probability of fraud

# ========== 6. Evaluation ==========
print("\n" + "="*50)
print("RANDOM FOREST RESULTS")
print("="*50)
print(classification_report(y_test, y_pred_rf, target_names=['Legit', 'Fraud']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_rf):.4f}")

print("\n" + "="*50)
print("ISOLATION FOREST RESULTS (using -1 as fraud)")
print("="*50)
# Convert Isolation Forest output: -1 = anomaly (fraud), 1 = normal
y_pred_iso_binary = (y_pred_iso == -1).astype(int)
print(classification_report(y_test, y_pred_iso_binary, target_names=['Legit', 'Fraud']))

# Feature importance from Random Forest
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 5 most important features:")
print(feature_importance.head())

# ========== 7. Save models and preprocessing objects ==========
model_dir = 'src/ml/models'
os.makedirs(model_dir, exist_ok=True)

joblib.dump(rf, f'{model_dir}/random_forest.pkl')
joblib.dump(iso_forest, f'{model_dir}/isolation_forest.pkl')
joblib.dump(scaler, f'{model_dir}/scaler.pkl')
joblib.dump(le_channel, f'{model_dir}/label_encoder_channel.pkl')

print(f"\n✅ Models saved to {model_dir}/")
print("Files created:")
print("  - random_forest.pkl")
print("  - isolation_forest.pkl")
print("  - scaler.pkl")
print("  - label_encoder_channel.pkl")
