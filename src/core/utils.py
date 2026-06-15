def format_timestamp(seconds: float) -> str:
    """Chuyển số giây thành dạng mm:ss.
    Ví dụ: 90.0 -> '1:30' | 5.0 -> '0:05' | 225.32 -> '3:45'
    """
    if seconds is None:
        return "?"
    seconds = float(seconds)
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"
