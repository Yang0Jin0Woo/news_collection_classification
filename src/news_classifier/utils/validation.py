class ValidationError(ValueError):
    pass


def require_non_blank(value: str, field_name: str) -> str:
    if value is None or not str(value).strip():
        raise ValidationError(f"{field_name} is required")
    return str(value).strip()


def validate_limit(limit: int, minimum: int = 1, maximum: int = 100) -> int:
    if not minimum <= limit <= maximum:
        raise ValidationError(
            f"limit must be between {minimum} and {maximum}"
        )
    return limit
