import numpy as np
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two coordinates"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

class FeatureExtractor:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def get_user_profile(self, user_id):
        """Fetch user's historical baseline from DB"""
        # In production, query user_behavioural_profiles table
        # For now, mock
        return {
            'avg_amount': 2500.0,
            'std_amount': 800.0,
            'typical_lat': 6.5244,
            'typical_lon': 3.3792,
            'typical_hour': 14,
            'channel_distribution': {'mobile_app': 0.7, 'ussd': 0.25, 'web': 0.05}
        }
    
    def extract_features(self, transaction, user_profile, recent_txns):
        """
        transaction: dict with amount, time, location, channel, device_id
        user_profile: dict from DB
        recent_txns: list of last N transactions for this user (for time-based features)
        """
        features = {}
        
        # Amount features
        amount = transaction['amount']
        features['amount'] = amount
        features['amount_z_score'] = (amount - user_profile['avg_amount']) / max(user_profile['std_amount'], 0.01)
        features['amount_ratio_to_avg'] = amount / max(user_profile['avg_amount'], 0.01)
        
        # Time features
        txn_time = transaction['transaction_time']
        features['hour_of_day'] = txn_time.hour
        features['day_of_week'] = txn_time.weekday()
        features['is_weekend'] = 1 if txn_time.weekday() >= 5 else 0
        features['is_night'] = 1 if (txn_time.hour < 5 or txn_time.hour > 22) else 0
        features['hour_deviation'] = abs(txn_time.hour - user_profile['typical_hour'])
        
        # Recency features
        if recent_txns:
            last_txn_time = recent_txns[0]['transaction_time']
            features['hours_since_last_txn'] = (txn_time - last_txn_time).total_seconds() / 3600
            features['days_since_last_txn'] = features['hours_since_last_txn'] / 24
            # Frequency in last hour
            last_hour = txn_time - timedelta(hours=1)
            features['txn_count_last_hour'] = sum(1 for t in recent_txns if t['transaction_time'] >= last_hour)
            features['txn_count_last_24h'] = sum(1 for t in recent_txns if (txn_time - t['transaction_time']).total_seconds() <= 86400)
        else:
            features['hours_since_last_txn'] = 9999
            features['days_since_last_txn'] = 9999
            features['txn_count_last_hour'] = 0
            features['txn_count_last_24h'] = 0
        
        # Location features
        if user_profile.get('typical_lat'):
            distance = haversine(
                transaction['location_lat'], transaction['location_lon'],
                user_profile['typical_lat'], user_profile['typical_lon']
            )
            features['distance_km'] = distance
            features['location_anomaly'] = 1 if distance > 50 else 0  # >50km from typical
        else:
            features['distance_km'] = 0
            features['location_anomaly'] = 0
        
        # Channel & device features
        features['channel'] = transaction['channel']
        # One-hot for ML (convert later)
        expected_channel = max(user_profile['channel_distribution'], key=user_profile['channel_distribution'].get)
        features['channel_change'] = 1 if transaction['channel'] != expected_channel else 0
        
        # Device feature (requires tracking previous devices per user)
        # Simplified: if device ID not in user's historical devices, flag
        # We'll implement this in the service layer
        
        return features
