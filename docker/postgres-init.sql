-- Create auxiliary databases (runs once on first postgres init)
SELECT 'CREATE DATABASE langfuse_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'langfuse_db')\gexec

SELECT 'CREATE DATABASE mlflow_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow_db')\gexec
