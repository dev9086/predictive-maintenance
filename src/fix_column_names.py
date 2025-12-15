"""
Fix column name mismatches between code and database
Run this to align your codebase with the database schema
"""
from db_connect import execute_query, fetch_dataframe
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_schema():
    """Check actual column names in model_predictions table"""
    query = """
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'model_predictions'
        ORDER BY ordinal_position
    """
    try:
        df = fetch_dataframe(query)
        logger.info("Current model_predictions schema:")
        print(df.to_string())
        return df
    except Exception as e:
        logger.error(f"Error checking schema: {e}")
        return None

def fix_schema_option_1():
    """Option 1: Rename database columns to match code"""
    statements = [
        "ALTER TABLE model_predictions RENAME COLUMN ts TO prediction_timestamp",
        "ALTER TABLE model_predictions RENAME COLUMN failure_prob TO failure_probability",
    ]
    
    logger.info("Option 1: Renaming database columns...")
    for stmt in statements:
        try:
            execute_query(stmt)
            logger.info(f"✓ {stmt}")
        except Exception as e:
            logger.error(f"✗ {stmt}: {e}")

def verify_fix():
    """Verify the fix worked"""
    query = "SELECT * FROM model_predictions LIMIT 1"
    try:
        df = fetch_dataframe(query)
        logger.info("✅ Verification successful. Columns:")
        print(df.columns.tolist())
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("Database Schema Fix Tool")
    logger.info("="*60)
    
    # Step 1: Check current schema
    schema_df = check_schema()
    
    if schema_df is not None:
        print("\n" + "="*60)
        print("Choose fix option:")
        print("1. Rename database columns (recommended)")
        print("2. Update code to match database (alternative)")
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            fix_schema_option_1()
            verify_fix()
            logger.info("\n✅ Database updated. Now update seed_model_predictions.py:")
            logger.info("   Change: ['machine_id', 'ts', 'failure_prob', ...]")
            logger.info("   To:     ['machine_id', 'prediction_timestamp', 'failure_probability', ...]")
        elif choice == "2":
            logger.info("\nUpdate streamlit_app.py and fastapi_server.py:")
            logger.info("  Replace 'prediction_timestamp' with 'ts'")
            logger.info("  Replace 'failure_probability' with 'failure_prob'")
    else:
        logger.error("Could not check schema. Check database connection.")
