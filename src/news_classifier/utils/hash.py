import hashlib
from news_classifier.utils.text import normalize_key


def stable_hash(*values: str) -> str:
    joined = "|".join(normalize_key(value) for value in values)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()
