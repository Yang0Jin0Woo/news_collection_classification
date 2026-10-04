from types import SimpleNamespace
import json

import pytest

from scripts import compare_body_enrichment as replay
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.collectors.article_scraper import ArticleScraper
from news_classifier.models import ModelPrediction
from news_classifier.rules.default_rules import DEFAULT_RULE_SET


def fake_pipeline():
    calls = []
    def classify(texts):
        calls.append(texts)
        return [ModelPrediction("교육/취업", .8, .2) if "기사본문:" in text
                else ModelPrediction("교육/취업", .2, .01) for text in texts]
    body = "학생 교육과 인재 양성을 위한 구체적인 교육 과정과 사업 운영 계획을 발표했습니다."
    return SimpleNamespace(
        classifier=SimpleNamespace(classify_many=classify),
        postprocessor=ClassificationPostProcessor(RuleEngine(DEFAULT_RULE_SET)),
        article_scraper=ArticleScraper(SimpleNamespace(get=lambda _: SimpleNamespace(
            text=f'<article class="fontsize-container">메뉴</article><div class="article-body">{body}</div>'))),
    ), calls


@pytest.mark.parametrize("keyword", ["AI", "지역", "스포츠", "앞으로 입력할 검색어"])
def test_same_articles_replay_is_query_independent_and_does_not_activate_candidates(keyword):
    pipeline, calls = fake_pipeline()
    rows = [{"title": "지역 소식", "link": "https://publisher.example/news/1", "keyword": keyword,
             "final_category": "검토필요"},
            {"title": "다른 지역 소식", "link": "https://publisher.example/news/2", "keyword": keyword,
             "content": "기존에 확보한 자세한 기사 본문 정보", "final_category": "교육/취업"}]
    before = json.dumps(rows)
    report = replay.compare_rows(rows, pipeline)
    assert report["before_review_count"] == 1 and report["after_review_count"] == 0
    assert report["previous_automatic_label_changes"] == 0
    assert len(calls) == 2 and len(calls[1]) == 1
    assert report["cases"][0]["extraction_diagnostics"]["method"] == "BODY_CONTAINER_BLOCKS"
    assert all(not c["activation_eligible"] for c in report["pending_rule_candidates"])
    assert all(case["gold_label"] == "" and case["review_status"] == "pending" for case in report["cases"])
    assert json.dumps(rows) == before


@pytest.mark.parametrize("prediction", [[], [ModelPrediction.failed()]])
def test_replay_rejects_baseline_model_failures(prediction):
    pipeline, _ = fake_pipeline()
    pipeline.classifier.classify_many = lambda _: prediction
    with pytest.raises(ValueError, match="baseline replay model failed"):
        replay.compare_rows([{"title": "news", "link": "https://publisher.example/1"}], pipeline)


def test_replay_does_not_overwrite_existing_report_or_source(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("keep", encoding="utf-8")
    with pytest.raises(SystemExit):
        replay.main(["--input", str(path), "--output", str(path)])
    assert path.read_text(encoding="utf-8") == "keep"


def test_replay_output_has_no_invented_gold_and_original_csv_is_preserved(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    source.write_text('title,link,final_category\nnews,https://publisher.example/1,검토필요\n', encoding="utf-8-sig")
    original = source.read_bytes()
    output = tmp_path / "report.json"
    monkeypatch.setattr(replay, "build_pipeline", lambda _: fake_pipeline()[0])
    replay.main(["--input", str(source), "--output", str(output)])
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert not payload["human_gold_confirmed"] and not payload["automatic_rule_activation"]
    assert payload["cases"][0]["gold_label"] == ""
    assert source.read_bytes() == original
