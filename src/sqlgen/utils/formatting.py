from typing import Any

def format_number_for_prompt(value: Any) -> str:
    """Format numeric values for readability while keeping the exact figure."""
    try:
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, float)):
            abs_val = abs(value)
            if abs_val >= 1_000_000_000_000:
                return f"{value:,.0f} ({value/1_000_000_000_000:.2f}T)"
            if abs_val >= 1_000_000_000:
                return f"{value:,.0f} ({value/1_000_000_000:.2f}B)"
            if abs_val >= 1_000_000:
                return f"{value:,.0f} ({value/1_000_000:.2f}M)"
            if 0 < abs_val < 0.001:
                return f"{value:.6f}"
            if isinstance(value, float):
                return f"{value:,.4f}".rstrip('0').rstrip('.')
            return f"{value:,}"
    except Exception:
        pass
    return str(value)

