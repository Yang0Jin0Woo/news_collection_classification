from news_classifier.classifiers.zero_shot_classifier import ZeroShotNewsClassifier


def test_classifier_loads_fixed_model_revision(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "news_classifier.classifiers.zero_shot_classifier.pipeline",
        fake_pipeline,
    )
    monkeypatch.setattr(
        "news_classifier.classifiers.zero_shot_classifier.torch.cuda.is_available",
        lambda: False,
    )
    classifier = ZeroShotNewsClassifier(
        model_name="model-name",
        model_revision="fixed-revision",
        candidate_labels=["기술개발"],
    )

    assert classifier._load() is sentinel
    assert captured["model"] == "model-name"
    assert captured["revision"] == "fixed-revision"
