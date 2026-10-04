"""합성 기사와 예측으로 조건부 재판단 검증. 실제 정확도 평가 아님."""
from dataclasses import replace
import json

import pytest

from news_classifier.classifiers.base import NewsClassifier
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.collectors.article_scraper import ArticleScraper, ArticleEnrichmentResult
from news_classifier.collectors.base import NewsCollector
from news_classifier.dedup.title_deduplicator import TitleSourceDeduplicator
from news_classifier.models import ModelPrediction, NewsItem, NewsReference, PipelineStatus
from news_classifier.pipeline import NewsPipeline
from news_classifier.rules.default_rules import DEFAULT_RULE_SET
from news_classifier.storage.csv_store import CsvNewsStore
from news_classifier.storage.sqlite_store import SqliteNewsStore
from news_classifier.rules.policy import RuleDecisionPolicy
from news_classifier.utils.http import HttpFetchResult
from types import SimpleNamespace


class Collector(NewsCollector):
    def __init__(self, *, content="", link="https://example.com/article"):
        self.content, self.link = content, link

    def fetch(self, keyword, limit):
        return [NewsItem(keyword, "지역의 새 소식", self.link, content=self.content)]


class Classifier(NewsClassifier):
    def __init__(self, *, initial=None, revised=None, failure=None):
        self.initial = initial or ModelPrediction("교육/취업", .3, .01)
        self.revised = revised or ModelPrediction("교육/취업", .85, .3)
        self.failure = failure
        self.calls = []

    def classify(self, text):
        return self.classify_many([text])[0]

    def classify_many(self, texts):
        self.calls.append(texts)
        if len(self.calls) == 1:
            return [self.initial for _ in texts]
        if self.failure == "exception":
            raise RuntimeError("synthetic model failure")
        if self.failure == "count":
            return []
        if self.failure == "failed":
            return [ModelPrediction.failed() for _ in texts]
        return [self.revised for _ in texts]


class Scraper:
    def __init__(self, *, content="학교 입학 상담과 신입생 교육 행사에 대한 자세한 기사 내용", fails=False):
        self.content, self.fails = content, fails
        self.calls = []

    def enrich(self, item):
        self.calls.append(item)
        if self.fails:
            raise RuntimeError("synthetic fetch failure")
        return replace(item, content=self.content)

    def enrich_many(self, items):
        return [self.enrich(item) for item in items]


def make_pipeline(*, collector=None, classifier=None, scraper=None, enabled=True):
    return NewsPipeline(
        collector or Collector(), classifier or Classifier(),
        ClassificationPostProcessor(RuleEngine(DEFAULT_RULE_SET)),
        TitleSourceDeduplicator(), scraper, enabled,
    )


@pytest.mark.parametrize("keyword", ["자동차", "AI", "스포츠", "임의의 새로운 검색어"])
def test_review_only_enrichment_is_keyword_independent(keyword):
    classifier, scraper = Classifier(), Scraper()
    result = make_pipeline(classifier=classifier, scraper=scraper).run(keyword)
    assert result.status == PipelineStatus.SUCCESS
    assert len(scraper.calls) == 1
    assert len(classifier.calls) == 2
    row = result.results[0]
    assert row.rule_decision.final_label == "교육/취업"
    assert row.item.keyword == keyword
    assert row.review_reclassification.initial_final_category == "검토필요"
    assert row.review_reclassification.initial_model_score == .3
    assert row.review_reclassification.status == "RECLASSIFIED"
    assert "검색주제:" not in classifier.calls[1][0]
    assert "기사본문:" in classifier.calls[1][0]


def test_confident_and_rule_decided_news_are_not_fetched():
    for prediction, collector in [
        (ModelPrediction("사회", .8, .2), Collector()),
        (ModelPrediction("교육/취업", .3, .01),
         type("RuleCollector", (Collector,), {"fetch": lambda self, keyword, limit: [
             NewsItem(keyword, "상장 첫날 따따블", "https://example.com/rule")
         ]})()),
    ]:
        classifier, scraper = Classifier(initial=prediction), Scraper()
        result = make_pipeline(collector=collector, classifier=classifier, scraper=scraper).run("공통")
        assert not result.results[0].review_required
        assert result.results[0].review_reclassification is None
        assert len(classifier.calls) == 1
        assert scraper.calls == []


@pytest.mark.parametrize("content,fails,status", [
    ("", False, "NO_CONTENT"), (" ", False, "NO_CONTENT"),
    ("본문", True, "FETCH_FAILED"),
])
def test_failed_content_enrichment_keeps_review(content, fails, status):
    classifier, scraper = Classifier(), Scraper(content=content, fails=fails)
    result = make_pipeline(classifier=classifier, scraper=scraper).run("공통")
    assert result.succeeded
    assert result.results[0].review_required
    assert result.results[0].review_reclassification.status == status
    assert len(classifier.calls) == 1


def test_still_uncertain_reclassification_is_not_forced_or_repeated():
    classifier = Classifier(revised=ModelPrediction("사회", .35, .01))
    scraper = Scraper()
    result = make_pipeline(classifier=classifier, scraper=scraper).run("공통")
    assert result.results[0].rule_decision.final_label == "검토필요"
    assert result.results[0].model_prediction.label == "사회"
    assert result.results[0].review_reclassification.status == "RECLASSIFIED"
    assert len(classifier.calls) == 2
    assert len(scraper.calls) == 1


@pytest.mark.parametrize("failure,code", [
    ("exception", "MODEL_INFERENCE_FAILURE"),
    ("count", "PREDICTION_COUNT_MISMATCH"),
    ("failed", "MODEL_PREDICTION_FAILURE"),
])
def test_retry_model_failure_remains_pipeline_failure(failure, code):
    result = make_pipeline(classifier=Classifier(failure=failure), scraper=Scraper()).run("공통")
    assert result.status == PipelineStatus.MODEL_ERROR
    assert not result.succeeded
    assert result.errors[0].stage == "RECLASSIFICATION"
    assert result.errors[0].code == code


@pytest.mark.parametrize("options,status", [
    ({"enabled": False, "scraper": Scraper()}, "DISABLED"),
    ({}, "NO_SCRAPER"),
    ({"collector": Collector(link=""), "scraper": Scraper()}, "NO_LINK"),
    ({"collector": Collector(content="기존에 확보한 기사 본문"), "scraper": Scraper()}, "CONTENT_PRESENT"),
])
def test_review_without_a_new_content_attempt(options, status):
    classifier = Classifier()
    result = make_pipeline(classifier=classifier, **options).run("공통")
    assert result.results[0].review_reclassification.status == status
    assert len(classifier.calls) == 1
    if options.get("scraper"):
        assert options["scraper"].calls == []


def test_explicit_full_enrichment_does_not_trigger_second_fetch():
    classifier, scraper = Classifier(), Scraper(content="")
    result = make_pipeline(classifier=classifier, scraper=scraper).run("공통", enrich_content=True)
    assert result.results[0].review_reclassification.status == "ALREADY_ENRICHED"
    assert len(scraper.calls) == 1
    assert len(classifier.calls) == 1


def test_same_effective_input_does_not_repeat_model_inference():
    classifier = Classifier()
    classifier.max_sequence_length = 10
    result = make_pipeline(classifier=classifier, scraper=Scraper()).run("공통")
    assert result.results[0].review_reclassification.status == "UNCHANGED_INPUT"
    assert len(classifier.calls) == 1


def test_explicit_enrichment_without_scraper_is_not_reported_as_attempted():
    result = make_pipeline().run("공통", enrich_content=True)
    assert result.results[0].review_reclassification.status == "NO_SCRAPER"


def test_group_identity_and_reclassification_audit_are_saved(tmp_path):
    class GroupCollector(Collector):
        def fetch(self, keyword, limit):
            return [replace(super().fetch(keyword, limit)[0], related_articles=(
                NewsReference("관련 기사", "다른 언론", "", "https://example.com/related"),
            ))]

    result = make_pipeline(collector=GroupCollector(), scraper=Scraper()).run("새 검색어")
    row = result.results[0]
    assert row.item.group_article_count == 2
    csv_store = CsvNewsStore(str(tmp_path / "result.csv"))
    csv_store.save([row])
    saved = csv_store.load().iloc[0]
    assert saved["group_article_count"] == 2
    assert json.loads(saved["review_reclassification"])["initial_final_category"] == "검토필요"
    sql_store = SqliteNewsStore(str(tmp_path / "result.db"))
    sql_store.save([row])
    with sql_store.connect() as connection:
        raw, count = connection.execute(
            "SELECT review_reclassification, group_article_count FROM classified_news"
        ).fetchone()
    assert count == 2
    assert json.loads(raw)["status"] == "RECLASSIFIED"


@pytest.mark.parametrize("mode,content,expected", [
    ("without_body", "", "교육/취업"),
    ("without_body", "새 본문", "검토필요"),
    ("with_body", "", "검토필요"),
    ("with_body", "새 본문", "교육/취업"),
])
def test_relaxed_policy_is_applied_only_to_its_validated_input_mode(mode, content, expected):
    # 운영 프로필의 성능을 주장하지 않고 합성 정책의 입력 범위 분기를 검증.
    relaxed = replace(DEFAULT_RULE_SET, decision=replace(
        RuleDecisionPolicy(), ambiguity_score=.4, ambiguity_margin=.01,
    ))
    processor = ClassificationPostProcessor(
        RuleEngine(relaxed), baseline_rule_engine=RuleEngine(DEFAULT_RULE_SET),
        decision_input_mode=mode,
    )
    result = processor.process(
        NewsItem("공통", "지역의 새 소식", "", content=content),
        ModelPrediction("교육/취업", .45, .03),
    )
    assert result.rule_decision.final_label == expected


def test_body_bound_profile_keeps_initial_review_then_retries_with_body():
    relaxed = replace(DEFAULT_RULE_SET, decision=replace(
        RuleDecisionPolicy(), ambiguity_score=.4, ambiguity_margin=.01,
    ))
    pipeline = make_pipeline(
        classifier=Classifier(
            initial=ModelPrediction("교육/취업", .45, .03),
            revised=ModelPrediction("교육/취업", .45, .03),
        ), scraper=Scraper(),
    )
    pipeline.postprocessor = ClassificationPostProcessor(
        RuleEngine(relaxed), baseline_rule_engine=RuleEngine(DEFAULT_RULE_SET),
        decision_input_mode="with_body",
    )
    result = pipeline.run("공통")
    assert result.results[0].rule_decision.final_label == "교육/취업"
    assert result.results[0].review_reclassification.initial_final_category == "검토필요"


@pytest.mark.parametrize("status,http_status", [
    ("TIMEOUT", None), ("BLOCKED_HTTP", 403), ("HTTP_ERROR", 404),
    ("REQUEST_FAILED", None),
])
def test_no_retry_http_failure_is_recorded_and_reported(status, http_status, caplog, capsys):
    classifier = Classifier()
    scraper = ArticleScraper(SimpleNamespace(
        get_with_diagnostics=lambda _: HttpFetchResult(None, status, http_status),
    ))
    with caplog.at_level("INFO"):
        result = make_pipeline(classifier=classifier, scraper=scraper).run("임의의 검색어")
    assert result.status == PipelineStatus.SUCCESS
    assert len(classifier.calls) == 1
    row = result.results[0]
    assert row.review_required
    audit = json.loads(row.to_row()["review_reclassification"])
    assert audit["status"] == status
    assert audit["enrichment_status"] == status
    assert audit["http_status"] == http_status
    assert audit["initial_final_category"] == "검토필요"
    assert "review enrichment outcomes" in caplog.text
    assert "검토 기사 본문 보강 결과:" in capsys.readouterr().out


@pytest.mark.parametrize("html,url,status", [
    ("<p>Generic sufficiently long wrapper notice should not be an article.</p>", "https://news.google.com/rss/articles/test", "GOOGLE_NEWS_WRAPPER"),
    ("<p>짧음</p>", "https://example.com/news", "NO_USABLE_PARAGRAPHS"),
])
def test_no_retry_page_failure_is_recorded(html, url, status):
    classifier = Classifier()
    scraper = ArticleScraper(SimpleNamespace(get=lambda _: SimpleNamespace(text=html, url=url)))
    result = make_pipeline(classifier=classifier, scraper=scraper).run("공통")
    audit = result.results[0].review_reclassification
    assert audit.status == status
    assert audit.enrichment_status == status
    assert len(classifier.calls) == 1


def test_explicit_full_enrichment_keeps_failure_diagnostics_without_second_fetch():
    calls = []
    def fetch(link):
        calls.append(link)
        return HttpFetchResult(None, "BLOCKED_HTTP", 429)
    classifier = Classifier()
    result = make_pipeline(classifier=classifier, scraper=ArticleScraper(SimpleNamespace(
        get_with_diagnostics=fetch,
    ))).run("공통", enrich_content=True)
    audit = result.results[0].review_reclassification
    assert audit.status == "ALREADY_ENRICHED"
    assert audit.enrichment_status == "BLOCKED_HTTP"
    assert audit.http_status == 429
    assert len(calls) == 1
    assert len(classifier.calls) == 1


def test_diagnostic_scraper_cannot_replace_item_or_group_identity():
    class DiagnosticScraper:
        def enrich_with_diagnostics(self, item):
            return ArticleEnrichmentResult(
                NewsItem("wrong", "wrong", "https://wrong.example/", content="새로운 기사 본문 정보"),
                "BODY_EXTRACTED", 200,
            )
    class GroupCollector(Collector):
        def fetch(self, keyword, limit):
            return [replace(super().fetch(keyword, limit)[0], related_articles=(
                NewsReference("관련 원문", "언론", "", "https://example.com/reference"),
            ))]
    classifier = Classifier()
    result = make_pipeline(collector=GroupCollector(), classifier=classifier, scraper=DiagnosticScraper()).run("임의 검색어")
    row = result.results[0]
    assert row.item.keyword == "임의 검색어"
    assert row.item.title == "지역의 새 소식"
    assert row.item.link == "https://example.com/article"
    assert row.item.group_article_count == 2
    assert row.review_reclassification.status == "RECLASSIFIED"
    assert row.review_reclassification.enrichment_status == "BODY_EXTRACTED"
    assert row.review_reclassification.http_status == 200
    assert len(classifier.calls) == 2


@pytest.mark.parametrize("blocked", [False, True])
def test_resolved_google_source_flows_through_review_retry_and_saved_audit(blocked):
    source = "https://news.google.com/rss/articles/CBMi" + "A" * 24
    publisher = "https://publisher.example/news"
    calls = []

    def fetch(url):
        calls.append(url)
        if url == publisher and blocked:
            return HttpFetchResult(None, "BLOCKED_HTTP", 403)
        html = (f'<link rel="canonical" href="{publisher}">' if url == source
                else "<article><p>학교 입학 상담과 신입생 교육 행사에 대한 충분히 긴 본문 내용으로 재판단에 필요한 상세 정보 제공.</p></article>")
        return HttpFetchResult(SimpleNamespace(text=html, url=url), "SUCCESS", 200)

    classifier = Classifier()
    scraper = ArticleScraper(SimpleNamespace(get_public_with_diagnostics=fetch))
    result = make_pipeline(collector=Collector(link=source), classifier=classifier, scraper=scraper).run("임의의 검색어")
    row = result.results[0]
    audit = json.loads(row.to_row()["review_reclassification"])
    assert row.item.link == source
    assert calls == [source, publisher]
    assert audit["resolved_url"] == publisher
    assert audit["resolution_status"] == "HTML_SOURCE_URL"
    assert audit["initial_model_score"] == .3
    assert audit["initial_final_category"] == "검토필요"
    if blocked:
        assert len(classifier.calls) == 1
        assert row.review_required
        assert audit["status"] == "BLOCKED_HTTP" and audit["http_status"] == 403
    else:
        assert len(classifier.calls) == 2
        assert not row.review_required
        assert audit["status"] == "RECLASSIFIED"
        assert audit["extraction_diagnostics"]["method"] == "ARTICLE_BLOCKS"
        assert audit["extraction_diagnostics"]["reason"] == "body_found"
