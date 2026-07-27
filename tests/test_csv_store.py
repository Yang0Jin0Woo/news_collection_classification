from news_classifier.models import CLASSIFIED_NEWS_COLUMNS
from news_classifier.storage.csv_store import CsvNewsStore


def test_empty_csv_contains_fixed_headers(tmp_path):
    path = tmp_path / "empty.csv"
    store = CsvNewsStore(str(path))

    store.save([])
    loaded = store.load()

    assert path.exists()
    assert loaded.empty
    assert list(loaded.columns) == CLASSIFIED_NEWS_COLUMNS
