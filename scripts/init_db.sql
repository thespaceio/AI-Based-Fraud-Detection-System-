-- 1. Users table
CREATE TABLE users (
    user_id UUID PRIMARY KEY,
    phone_number VARCHAR(15) NOT NULL UNIQUE,
    email VARCHAR(255),
    registration_date TIMESTAMPTZ NOT NULL,
    account_status VARCHAR(20) DEFAULT 'active',
    -- behavioural profile aggregates (denormalised for speed)
    avg_transaction_amount DECIMAL(12,2),
    avg_transaction_frequency_per_day DECIMAL(5,2),
    typical_location GEOGRAPHY(POINT),  -- PostGIS optional, else lat/lon
    typical_hour_of_day INT,
    last_updated TIMESTAMPTZ
);

-- 2. Transactions table
CREATE TABLE transactions (
    transaction_id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(user_id),
    amount DECIMAL(12,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'NGN',
    transaction_time TIMESTAMPTZ NOT NULL,
    location_lat DOUBLE PRECISION,
    location_lon DOUBLE PRECISION,
    device_id VARCHAR(100),
    channel VARCHAR(20) CHECK (channel IN ('mobile_app', 'ussd', 'web', 'pos', 'atm')),
    ip_address INET,
    recipient_account VARCHAR(50),
    risk_score SMALLINT,           -- 0-100
    is_fraud BOOLEAN DEFAULT NULL, -- labelled for training (offline)
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Fraud alerts
CREATE TABLE fraud_alerts (
    alert_id UUID PRIMARY KEY,
    transaction_id UUID REFERENCES transactions(transaction_id),
    user_id UUID REFERENCES users(user_id),
    alert_time TIMESTAMPTZ DEFAULT NOW(),
    risk_score SMALLINT NOT NULL,
    reason TEXT,                    -- explanation: "amount deviation + location anomaly"
    status VARCHAR(20) DEFAULT 'pending', -- pending, reviewed, false_positive
    reviewer_notes TEXT
);

-- 4. Behavioural profiles (time-series aggregations for quick retrieval)
CREATE TABLE user_behavioural_profiles (
    profile_id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(user_id),
    window_start DATE,              -- daily profile
    window_end DATE,
    avg_amount DECIMAL(12,2),
    std_amount DECIMAL(12,2),
    median_amount DECIMAL(12,2),
    transaction_count INT,
    most_frequent_hour INT,
    most_frequent_location GEOGRAPHY(POINT),
    -- JSONB for flexible features (e.g., channel usage)
    channel_frequency JSONB,        -- {"mobile_app": 0.8, "ussd": 0.2}
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_txn_user_time ON transactions(user_id, transaction_time DESC);
CREATE INDEX idx_txn_time ON transactions(transaction_time);
CREATE INDEX idx_alerts_status ON fraud_alerts(status);
CREATE INDEX idx_profile_user ON user_behavioural_profiles(user_id, window_start DESC);
