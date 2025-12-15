"""Initialize database schema and seed data from SQL files.

Reads sql/schema.sql and sql/seed_data.sql and executes them against the DB.

Run: python src/init_db.py
"""
import os
from db_connect import execute_query

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SCHEMA_FILE = os.path.join(BASE_DIR, 'sql', 'schema.sql')
SEED_FILE = os.path.join(BASE_DIR, 'sql', 'seed_data.sql')

def load_sql_file(filepath):
    """Read SQL file and split into individual statements."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    # Split on semicolon and filter empty statements
    statements = [s.strip() for s in content.split(';') if s.strip()]
    return statements

def main():
    print("Loading schema from", SCHEMA_FILE)
    schema_stmts = load_sql_file(SCHEMA_FILE)
    for stmt in schema_stmts:
        try:
            execute_query(stmt)
            print(f'✓ Executed statement')
        except Exception as e:
            print(f'✗ Failed: {e}')

    print("\nLoading seed data from", SEED_FILE)
    seed_stmts = load_sql_file(SEED_FILE)
    for stmt in seed_stmts:
        try:
            execute_query(stmt)
            print(f'✓ Executed statement')
        except Exception as e:
            print(f'✗ Failed: {e}')

    print("\n✅ Database initialization complete!")

if __name__ == '__main__':
    main()
