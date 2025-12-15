"""
Configuration Management for Predictive Maintenance System
Loads environment variables and provides centralized configuration
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for database and application settings"""
    
    # Database Configuration
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', 5433))
    DB_NAME = os.getenv('DB_NAME', 'zepto_sql_project')
    DB_USER = os.getenv('DB_USER', 'pm_user')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'khushi#2205')
    
    # API Configuration
    API_PORT = int(os.getenv('API_PORT', 5000))
    API_DEBUG = os.getenv('FLASK_ENV', 'development') == 'development'
    
    # Model Configuration
    MODEL_VERSION = os.getenv('MODEL_VERSION', 'v1.0')
    FAILURE_THRESHOLD = float(os.getenv('FAILURE_THRESHOLD', 0.7))
    RUL_ALERT_DAYS = int(os.getenv('RUL_ALERT_DAYS', 7))
    
    # Alert Configuration
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
    SMTP_USERNAME = os.getenv('SMTP_USERNAME', '')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')
    ALERT_EMAIL = os.getenv('ALERT_EMAIL', '')
    
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    @classmethod
    def get_db_connection_string(cls):
        """Return database connection string"""
        return f"postgresql://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
    
    @classmethod
    def validate(cls):
        """Validate that all required configurations are set"""
        required = ['DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
        missing = [key for key in required if not getattr(cls, key)]
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")
        return True

# Validate configuration on import
Config.validate()
