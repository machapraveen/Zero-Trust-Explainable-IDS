-- ZT-XAIIDS PostgreSQL Initialization Script
-- Explainable Deep Learning Framework for Zero-Trust Intrusion Detection

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

GRANT ALL PRIVILEGES ON DATABASE ztxai_ids TO ztxai;
