"""Drop all tables and reinitialize database schema and seed data from SQL files.

Run: python src/recreate_db.py
"""
import os
from db_connect import execute_query

def main():
    # Drop all tables in correct order (respecting foreign keys)
    drop_stmts = [
        "DROP TABLE IF EXISTS model_predictions CASCADE",
        "DROP TABLE IF EXISTS external_data CASCADE",
        "DROP TABLE IF EXISTS failure_logs CASCADE",
        "DROP TABLE IF EXISTS sensor_readings CASCADE",
        "DROP TABLE IF EXISTS machines CASCADE",
    ]
    
    print("Dropping existing tables...")
    for stmt in drop_stmts:
        try:
            execute_query(stmt)
            print(f'✓ {stmt}')
        except Exception as e:
            print(f'✗ Failed: {e}')

    # Now recreate schema
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    SCHEMA_FILE = os.path.join(BASE_DIR, 'sql', 'schema.sql')
    SEED_FILE = os.path.join(BASE_DIR, 'sql', 'seed_data.sql')

    print("\nLoading schema from", SCHEMA_FILE)
    with open(SCHEMA_FILE, 'r', encoding='utf-8') as f:
        schema_content = f.read()
    schema_stmts = [s.strip() for s in schema_content.split(';') if s.strip()]
    for stmt in schema_stmts:
        try:
            execute_query(stmt)
            print(f'✓ Created table/object')
        except Exception as e:
            print(f'✗ Failed: {e}')

    print("\nLoading seed data from", SEED_FILE)
    with open(SEED_FILE, 'r', encoding='utf-8') as f:
        seed_content = f.read()
    seed_stmts = [s.strip() for s in seed_content.split(';') if s.strip()]
    for stmt in seed_stmts:
        try:
            execute_query(stmt)
            print(f'✓ Inserted data')
        except Exception as e:
            print(f'✗ Failed: {e}')

    print("\n✅ Database recreated successfully!")

if __name__ == '__main__':
    main()
