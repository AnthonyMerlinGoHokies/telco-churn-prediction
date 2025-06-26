-- PostgreSQL setup for Telco Customer Churn Analysis

-- Create database (run this from command line)
-- createdb telco_churn

-- Connect to the database and create the customers table
\c telco_churn;

-- Drop table if exists (for fresh start)
DROP TABLE IF EXISTS customers;

-- Create the customers table
CREATE TABLE customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    gender VARCHAR(10),
    senior_citizen INTEGER,
    partner VARCHAR(5),
    dependents VARCHAR(5),
    tenure INTEGER,
    phone_service VARCHAR(5),
    multiple_lines VARCHAR(25),
    internet_service VARCHAR(25),
    online_security VARCHAR(25),
    online_backup VARCHAR(25),
    device_protection VARCHAR(25),
    tech_support VARCHAR(25),
    streaming_tv VARCHAR(25),
    streaming_movies VARCHAR(25),
    contract VARCHAR(25),
    paperless_billing VARCHAR(5),
    payment_method VARCHAR(50),
    monthly_charges DECIMAL(10,2),
    total_charges DECIMAL(10,2),
    churn VARCHAR(5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX idx_customers_churn ON customers(churn);
CREATE INDEX idx_customers_contract ON customers(contract);
CREATE INDEX idx_customers_tenure ON customers(tenure);
CREATE INDEX idx_customers_monthly_charges ON customers(monthly_charges);

-- Verify table creation
\d customers;

COMMENT ON TABLE customers IS 'Telco customer data for churn prediction analysis';
COMMENT ON COLUMN customers.customer_id IS 'Unique identifier for each customer';
COMMENT ON COLUMN customers.churn IS 'Target variable: Yes/No indicating if customer churned';
COMMENT ON COLUMN customers.tenure IS 'Number of months customer has stayed with company';
COMMENT ON COLUMN customers.monthly_charges IS 'Amount charged to customer monthly';
COMMENT ON COLUMN customers.total_charges IS 'Total amount charged to customer';
