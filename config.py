"""
Configuration file for Hindi OCR Web Application
"""

import os

# Flask Configuration
class Config:
    """Base configuration"""
    DEBUG = False
    TESTING = False
    
    # Upload settings
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    
    # Model settings
    MODEL_DIR = os.getenv('MODEL_DIR', 'model_hindi')
    MODEL_REPO_ID = 'rajesh-1902/hindi-ocr-crnn-ctc-v2'
    
    # Logging
    LOG_LEVEL = 'INFO'


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    UPLOAD_FOLDER = 'test_uploads'


# Load configuration based on environment
ENV = os.getenv('FLASK_ENV', 'development')

if ENV == 'production':
    app_config = ProductionConfig()
elif ENV == 'testing':
    app_config = TestingConfig()
else:
    app_config = DevelopmentConfig()
