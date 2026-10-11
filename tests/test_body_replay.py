from types import SimpleNamespace
from dataclasses import replace
import json

import pytest

from scripts import compare_body_enrichment as replay
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.collectors.article_scraper import ArticleEnrichmentResult, ArticleScraper
from news_classifier.dedup.title_deduplicator import TitleSourceDeduplicator
from news_classifier.models import ModelPrediction, NewsItem
from news_classifier.pipeline import NewsPipeline
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


@pytest.mark.parametrize("limit,initial_content,new_content,expected_retries", [
    (10, "", "새로운 상세 기사 본문", 0),
    (1200, "", "새로운 상세 기사 본문", 1),
    (1200, "", "&nbsp;", 0),
    (1200, "&nbsp;", "새로운 상세 기사 본문", 1),
])
def test_replay_and_runtime_share_effective_input_retry_condition(
    limit, initial_content, new_content, expected_retries,
):
    row = {
        "keyword": "임의 검색어", "title": "지역 소식",
        "link": "https://example.com/news", "content": initial_content,
        "final_category": "검토필요",
    }
    item = NewsItem(**{key: row[key] for key in ("keyword", "title", "link", "content")})

    def dependencies():
        calls = []

        def classify(texts):
            calls.append(texts)
            prediction = ModelPrediction("사회", .2, .01) if len(calls) == 1 else ModelPrediction("사회", .8, .2)
            return [prediction for _ in texts]

        return SimpleNamespace(
            classifier=SimpleNamespace(classify_many=classify, max_sequence_length=limit),
            postprocessor=ClassificationPostProcessor(RuleEngine(DEFAULT_RULE_SET)),
            article_scraper=SimpleNamespace(enrich_with_diagnostics=lambda original: ArticleEnrichmentResult(
                replace(original, content=new_content), "BODY_EXTRACTED", 200,
            )),
        ), calls

    replay_pipeline, replay_calls = dependencies()
    runtime_dependencies, runtime_calls = dependencies()
    result = NewsPipeline(
        collector=SimpleNamespace(fetch=lambda **_: [item]),
        classifier=runtime_dependencies.classifier,
        postprocessor=runtime_dependencies.postprocessor,
        deduplicator=TitleSourceDeduplicator(),
        article_scraper=runtime_dependencies.article_scraper,
    ).run(row["keyword"])
    report = replay.compare_rows([row], replay_pipeline)

    assert result.succeeded
    assert len(replay_calls) == len(runtime_calls) == 1 + expected_retries
    assert report["reclassified_count"] == expected_retries
    assert report["cases"][0]["reclassified"] is bool(expected_retries)
    assert report["after_category_counts"] == {result.results[0].rule_decision.final_label: 1}
    if limit == 10:
        assert result.results[0].review_reclassification.status == "UNCHANGED_INPUT"


@pytest.mark.parametrize("raw", [
    "title,link,final_category,title\nnews,url,검토필요,other\n",
    "title,link,final_category\nnews,url\n",
    "title,link,final_category\nnews,url,검토필요,extra\n",
])
def test_replay_rejects_bad_csv_before_model_loading(tmp_path, monkeypatch, raw):
    source, output = tmp_path / "source.csv", tmp_path / "report.json"
    source.write_text(raw, encoding="utf-8-sig")
    original = source.read_bytes()
    monkeypatch.setattr(replay, "build_pipeline", lambda _: pytest.fail("invalid input must not load the model"))
    with pytest.raises(SystemExit) as exc:
        replay.main(["--input", str(source), "--output", str(output)])
    assert exc.value.code == 2
    assert source.read_bytes() == original
    assert not output.exists()


def test_replay_aborts_report_when_source_changes_during_comparison(tmp_path, monkeypatch):
    source, output = tmp_path / "source.csv", tmp_path / "report.json"
    source.write_text("title,link,final_category\nnews,url,검토필요\n", encoding="utf-8-sig")
    monkeypatch.setattr(replay, "build_pipeline", lambda _: fake_pipeline()[0])

    def change_source(rows, pipeline):
        source.write_text("changed by another task", encoding="utf-8")
        return {}

    monkeypatch.setattr(replay, "compare_rows", change_source)
    with pytest.raises(SystemExit) as exc:
        replay.main(["--input", str(source), "--output", str(output)])
    assert exc.value.code == 2
    assert not output.exists()
