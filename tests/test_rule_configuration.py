import pytest

from news_classifier.rules.default_rules import validate_rule_configuration


def test_default_rule_configuration_is_valid():
    validate_rule_configuration()


def test_rule_configuration_rejects_label_mismatch():
    with pytest.raises(ValueError, match="rule labels mismatch"):
        validate_rule_configuration(
            candidate_labels=["기술개발", "제품/서비스"],
            rules={"기술개발": ["연구"]},
        )


def test_rule_configuration_rejects_keyword_shared_by_labels():
    with pytest.raises(ValueError, match="multiple labels"):
        validate_rule_configuration(
            candidate_labels=["기술개발", "제품/서비스"],
            rules={
                "기술개발": ["AI"],
                "제품/서비스": ["ai"],
            },
        )
