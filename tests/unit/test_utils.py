import pytest
from src.core.utils import format_timestamp

def test_format_timestamp():
    """Test format_timestamp behavior."""
    # Test None
    assert format_timestamp(None) == "?"
    
    # Test under 60 seconds
    assert format_timestamp(5.24) == "5.2s"
    assert format_timestamp(0.0) == "0.0s"
    
    # Test exactly 60 seconds
    assert format_timestamp(60.0) == "1:00"
    
    # Test over 60 seconds
    assert format_timestamp(90.0) == "1:30"
    assert format_timestamp(225.32) == "3:45"
    
    # Test string input that can be cast to float
    assert format_timestamp("5.24") == "5.2s"
    assert format_timestamp("90") == "1:30"
    
    # Test invalid string (should raise ValueError)
    with pytest.raises(ValueError):
        format_timestamp("abc")
