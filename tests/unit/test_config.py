from pydantic import SecretStr
from src.core.config import Settings

def test_settings_initialization():
    """Kiểm tra khởi tạo Settings không bị lỗi."""
    # Khởi tạo Settings, chấp nhận các giá trị từ .env local
    settings = Settings()
    
    assert isinstance(settings.PROJECT_NAME, str)
    assert isinstance(settings.VERSION, str)
    assert settings.ENVIRONMENT in ["development", "staging", "production"]
    assert hasattr(settings, "LLM_PROVIDER")
    assert hasattr(settings, "is_production")

def test_settings_is_production(monkeypatch):
    """Kiểm tra logic @computed_field is_production."""
    # Bơm thử biến môi trường ENVIRONMENT
    monkeypatch.setenv("ENVIRONMENT", "production")
    settings = Settings(_env_file=None)
    
    assert settings.ENVIRONMENT == "production"
    assert settings.is_production is True

def test_settings_secret_keys(monkeypatch):
    """Kiểm tra tính năng che giấu SecretStr."""
    test_key = "gsk_test_key_123"
    monkeypatch.setenv("GROQ_API_KEY", test_key)
    
    settings = Settings(_env_file=None)
    
    # GROQ_API_KEY phải được tự động parse thành SecretStr
    assert isinstance(settings.GROQ_API_KEY, SecretStr)
    assert settings.GROQ_API_KEY.get_secret_value() == test_key
    # Đảm bảo khi print không bị leak key
    assert test_key not in str(settings.GROQ_API_KEY)
