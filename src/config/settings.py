import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BaseConfig:
    """Base Configuration"""
    # Secrets (Must be loaded from environment)
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    
    # Defaults
    CAMERA_SOURCE = 0
    LOG_FILE = os.path.join("logs", "activity_log.csv")
    MODEL_PERSON = "yolov8n.pt"
    MODEL_PPE = "hardhat.pt"
    MAX_HISTORY = 64
    
    # Safety Check
    @classmethod
    def validate(cls):
        missing = []
        if not cls.TELEGRAM_TOKEN: missing.append("TELEGRAM_BOT_TOKEN")
        if not cls.TELEGRAM_CHAT_ID: missing.append("TELEGRAM_CHAT_ID")
        
        if missing:
            print(f"Warning: Missing environment variables: {', '.join(missing)}")

class DevelopmentConfig(BaseConfig):
    """Development Settings"""
    DEBUG = True
    # Relaxed thresholds for testing
    CONF_PERSON = 0.4
    CONF_HELMET = 0.5
    ALERT_COOLDOWN = 5 # 5 seconds for easier debugging

class ProductionConfig(BaseConfig):
    """Production Settings"""
    DEBUG = False
    # Stricter thresholds for fewer false alarms
    CONF_PERSON = 0.6
    CONF_HELMET = 0.7
    ALERT_COOLDOWN = 30 # 30 seconds to prevent spam
    
# Configuration Factory
def get_config():
    env = os.getenv("APP_ENV", "development").lower()
    
    if env == "production":
        return ProductionConfig
    return DevelopmentConfig

# Export the active configuration
Config = get_config()
Config.validate()
