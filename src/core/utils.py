def format_timestamp(seconds: float) -> str:
    """Chuyển số giây thành dạng mm:ss nếu >= 60 giây, ngược lại giữ nguyên dạng Xs.
    Ví dụ: 90.0 -> '1:30' | 5.24 -> '5.2s' | 225.32 -> '3:45'
    """
    if seconds is None:
        return "?"
    seconds = float(seconds)
    if seconds >= 60:
        m = int(seconds) // 60
        s = int(seconds) % 60
        return f"{m}:{s:02d}"
    else:
        return f"{seconds:.1f}s"
