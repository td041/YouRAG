import pytest
from src.core.utils import format_timestamp

def test_format_timestamp():
    """Test format_timestamp behavior — always mm:ss."""
    assert format_timestamp(None) == "?"

    # Under 60 seconds → 0:ss
    assert format_timestamp(5.24) == "0:05"
    assert format_timestamp(0.0) == "0:00"
    assert format_timestamp(3.0) == "0:03"

    # Exactly 60 seconds
    assert format_timestamp(60.0) == "1:00"

    # Over 60 seconds
    assert format_timestamp(90.0) == "1:30"
    assert format_timestamp(225.32) == "3:45"

    # String input
    assert format_timestamp("5.24") == "0:05"
    assert format_timestamp("90") == "1:30"

    # Invalid string
    with pytest.raises(ValueError):
        format_timestamp("abc")
