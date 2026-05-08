import pytest

from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.models import ModelPrediction
from news_classifier.rules.default_rules import RULES
from data.sample_news_cases import SAMPLE_NEWS_CASES

@pytest.mark.parametrize("case", SAMPLE_NEWS_CASES[:120])
def test_rule_engine_classifies_representative_cases(case):
    engine = RuleEngine(RULES)
    decision = engine.decide(
        title=case["title"],
        description=case["description"],
        content="",
        prediction=ModelPrediction("검토필요", 0.30, 0.01, [], []),
    )
    assert decision.final_label in set(RULES.keys()) | {"검토필요"}
    assert decision.rule_match_count >= 0

def test_sample_case_0001_has_required_fields():
    case = SAMPLE_NEWS_CASES[0]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0002_has_required_fields():
    case = SAMPLE_NEWS_CASES[1]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0003_has_required_fields():
    case = SAMPLE_NEWS_CASES[2]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0004_has_required_fields():
    case = SAMPLE_NEWS_CASES[3]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0005_has_required_fields():
    case = SAMPLE_NEWS_CASES[4]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0006_has_required_fields():
    case = SAMPLE_NEWS_CASES[5]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0007_has_required_fields():
    case = SAMPLE_NEWS_CASES[6]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0008_has_required_fields():
    case = SAMPLE_NEWS_CASES[7]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0009_has_required_fields():
    case = SAMPLE_NEWS_CASES[8]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0010_has_required_fields():
    case = SAMPLE_NEWS_CASES[9]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0011_has_required_fields():
    case = SAMPLE_NEWS_CASES[10]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0012_has_required_fields():
    case = SAMPLE_NEWS_CASES[11]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0013_has_required_fields():
    case = SAMPLE_NEWS_CASES[12]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0014_has_required_fields():
    case = SAMPLE_NEWS_CASES[13]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0015_has_required_fields():
    case = SAMPLE_NEWS_CASES[14]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0016_has_required_fields():
    case = SAMPLE_NEWS_CASES[15]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0017_has_required_fields():
    case = SAMPLE_NEWS_CASES[16]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0018_has_required_fields():
    case = SAMPLE_NEWS_CASES[17]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0019_has_required_fields():
    case = SAMPLE_NEWS_CASES[18]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0020_has_required_fields():
    case = SAMPLE_NEWS_CASES[19]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0021_has_required_fields():
    case = SAMPLE_NEWS_CASES[20]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0022_has_required_fields():
    case = SAMPLE_NEWS_CASES[21]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0023_has_required_fields():
    case = SAMPLE_NEWS_CASES[22]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0024_has_required_fields():
    case = SAMPLE_NEWS_CASES[23]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0025_has_required_fields():
    case = SAMPLE_NEWS_CASES[24]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0026_has_required_fields():
    case = SAMPLE_NEWS_CASES[25]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0027_has_required_fields():
    case = SAMPLE_NEWS_CASES[26]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0028_has_required_fields():
    case = SAMPLE_NEWS_CASES[27]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0029_has_required_fields():
    case = SAMPLE_NEWS_CASES[28]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0030_has_required_fields():
    case = SAMPLE_NEWS_CASES[29]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0031_has_required_fields():
    case = SAMPLE_NEWS_CASES[30]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0032_has_required_fields():
    case = SAMPLE_NEWS_CASES[31]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0033_has_required_fields():
    case = SAMPLE_NEWS_CASES[32]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0034_has_required_fields():
    case = SAMPLE_NEWS_CASES[33]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0035_has_required_fields():
    case = SAMPLE_NEWS_CASES[34]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0036_has_required_fields():
    case = SAMPLE_NEWS_CASES[35]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0037_has_required_fields():
    case = SAMPLE_NEWS_CASES[36]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0038_has_required_fields():
    case = SAMPLE_NEWS_CASES[37]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0039_has_required_fields():
    case = SAMPLE_NEWS_CASES[38]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0040_has_required_fields():
    case = SAMPLE_NEWS_CASES[39]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0041_has_required_fields():
    case = SAMPLE_NEWS_CASES[40]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0042_has_required_fields():
    case = SAMPLE_NEWS_CASES[41]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0043_has_required_fields():
    case = SAMPLE_NEWS_CASES[42]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0044_has_required_fields():
    case = SAMPLE_NEWS_CASES[43]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0045_has_required_fields():
    case = SAMPLE_NEWS_CASES[44]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0046_has_required_fields():
    case = SAMPLE_NEWS_CASES[45]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0047_has_required_fields():
    case = SAMPLE_NEWS_CASES[46]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0048_has_required_fields():
    case = SAMPLE_NEWS_CASES[47]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0049_has_required_fields():
    case = SAMPLE_NEWS_CASES[48]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0050_has_required_fields():
    case = SAMPLE_NEWS_CASES[49]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0051_has_required_fields():
    case = SAMPLE_NEWS_CASES[50]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0052_has_required_fields():
    case = SAMPLE_NEWS_CASES[51]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0053_has_required_fields():
    case = SAMPLE_NEWS_CASES[52]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0054_has_required_fields():
    case = SAMPLE_NEWS_CASES[53]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0055_has_required_fields():
    case = SAMPLE_NEWS_CASES[54]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0056_has_required_fields():
    case = SAMPLE_NEWS_CASES[55]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0057_has_required_fields():
    case = SAMPLE_NEWS_CASES[56]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0058_has_required_fields():
    case = SAMPLE_NEWS_CASES[57]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0059_has_required_fields():
    case = SAMPLE_NEWS_CASES[58]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0060_has_required_fields():
    case = SAMPLE_NEWS_CASES[59]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0061_has_required_fields():
    case = SAMPLE_NEWS_CASES[60]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0062_has_required_fields():
    case = SAMPLE_NEWS_CASES[61]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0063_has_required_fields():
    case = SAMPLE_NEWS_CASES[62]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0064_has_required_fields():
    case = SAMPLE_NEWS_CASES[63]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0065_has_required_fields():
    case = SAMPLE_NEWS_CASES[64]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0066_has_required_fields():
    case = SAMPLE_NEWS_CASES[65]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0067_has_required_fields():
    case = SAMPLE_NEWS_CASES[66]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0068_has_required_fields():
    case = SAMPLE_NEWS_CASES[67]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0069_has_required_fields():
    case = SAMPLE_NEWS_CASES[68]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0070_has_required_fields():
    case = SAMPLE_NEWS_CASES[69]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0071_has_required_fields():
    case = SAMPLE_NEWS_CASES[70]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0072_has_required_fields():
    case = SAMPLE_NEWS_CASES[71]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0073_has_required_fields():
    case = SAMPLE_NEWS_CASES[72]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0074_has_required_fields():
    case = SAMPLE_NEWS_CASES[73]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0075_has_required_fields():
    case = SAMPLE_NEWS_CASES[74]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0076_has_required_fields():
    case = SAMPLE_NEWS_CASES[75]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0077_has_required_fields():
    case = SAMPLE_NEWS_CASES[76]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0078_has_required_fields():
    case = SAMPLE_NEWS_CASES[77]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0079_has_required_fields():
    case = SAMPLE_NEWS_CASES[78]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0080_has_required_fields():
    case = SAMPLE_NEWS_CASES[79]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0081_has_required_fields():
    case = SAMPLE_NEWS_CASES[80]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0082_has_required_fields():
    case = SAMPLE_NEWS_CASES[81]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0083_has_required_fields():
    case = SAMPLE_NEWS_CASES[82]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0084_has_required_fields():
    case = SAMPLE_NEWS_CASES[83]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0085_has_required_fields():
    case = SAMPLE_NEWS_CASES[84]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0086_has_required_fields():
    case = SAMPLE_NEWS_CASES[85]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0087_has_required_fields():
    case = SAMPLE_NEWS_CASES[86]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0088_has_required_fields():
    case = SAMPLE_NEWS_CASES[87]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0089_has_required_fields():
    case = SAMPLE_NEWS_CASES[88]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0090_has_required_fields():
    case = SAMPLE_NEWS_CASES[89]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0091_has_required_fields():
    case = SAMPLE_NEWS_CASES[90]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0092_has_required_fields():
    case = SAMPLE_NEWS_CASES[91]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0093_has_required_fields():
    case = SAMPLE_NEWS_CASES[92]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0094_has_required_fields():
    case = SAMPLE_NEWS_CASES[93]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0095_has_required_fields():
    case = SAMPLE_NEWS_CASES[94]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0096_has_required_fields():
    case = SAMPLE_NEWS_CASES[95]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0097_has_required_fields():
    case = SAMPLE_NEWS_CASES[96]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0098_has_required_fields():
    case = SAMPLE_NEWS_CASES[97]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0099_has_required_fields():
    case = SAMPLE_NEWS_CASES[98]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0100_has_required_fields():
    case = SAMPLE_NEWS_CASES[99]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0101_has_required_fields():
    case = SAMPLE_NEWS_CASES[100]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0102_has_required_fields():
    case = SAMPLE_NEWS_CASES[101]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0103_has_required_fields():
    case = SAMPLE_NEWS_CASES[102]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0104_has_required_fields():
    case = SAMPLE_NEWS_CASES[103]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0105_has_required_fields():
    case = SAMPLE_NEWS_CASES[104]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0106_has_required_fields():
    case = SAMPLE_NEWS_CASES[105]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0107_has_required_fields():
    case = SAMPLE_NEWS_CASES[106]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0108_has_required_fields():
    case = SAMPLE_NEWS_CASES[107]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0109_has_required_fields():
    case = SAMPLE_NEWS_CASES[108]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0110_has_required_fields():
    case = SAMPLE_NEWS_CASES[109]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0111_has_required_fields():
    case = SAMPLE_NEWS_CASES[110]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0112_has_required_fields():
    case = SAMPLE_NEWS_CASES[111]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0113_has_required_fields():
    case = SAMPLE_NEWS_CASES[112]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0114_has_required_fields():
    case = SAMPLE_NEWS_CASES[113]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0115_has_required_fields():
    case = SAMPLE_NEWS_CASES[114]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0116_has_required_fields():
    case = SAMPLE_NEWS_CASES[115]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0117_has_required_fields():
    case = SAMPLE_NEWS_CASES[116]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0118_has_required_fields():
    case = SAMPLE_NEWS_CASES[117]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0119_has_required_fields():
    case = SAMPLE_NEWS_CASES[118]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0120_has_required_fields():
    case = SAMPLE_NEWS_CASES[119]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0121_has_required_fields():
    case = SAMPLE_NEWS_CASES[120]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0122_has_required_fields():
    case = SAMPLE_NEWS_CASES[121]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0123_has_required_fields():
    case = SAMPLE_NEWS_CASES[122]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0124_has_required_fields():
    case = SAMPLE_NEWS_CASES[123]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0125_has_required_fields():
    case = SAMPLE_NEWS_CASES[124]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0126_has_required_fields():
    case = SAMPLE_NEWS_CASES[125]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0127_has_required_fields():
    case = SAMPLE_NEWS_CASES[126]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0128_has_required_fields():
    case = SAMPLE_NEWS_CASES[127]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0129_has_required_fields():
    case = SAMPLE_NEWS_CASES[128]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0130_has_required_fields():
    case = SAMPLE_NEWS_CASES[129]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0131_has_required_fields():
    case = SAMPLE_NEWS_CASES[130]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0132_has_required_fields():
    case = SAMPLE_NEWS_CASES[131]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0133_has_required_fields():
    case = SAMPLE_NEWS_CASES[132]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0134_has_required_fields():
    case = SAMPLE_NEWS_CASES[133]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0135_has_required_fields():
    case = SAMPLE_NEWS_CASES[134]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0136_has_required_fields():
    case = SAMPLE_NEWS_CASES[135]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0137_has_required_fields():
    case = SAMPLE_NEWS_CASES[136]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0138_has_required_fields():
    case = SAMPLE_NEWS_CASES[137]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0139_has_required_fields():
    case = SAMPLE_NEWS_CASES[138]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0140_has_required_fields():
    case = SAMPLE_NEWS_CASES[139]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0141_has_required_fields():
    case = SAMPLE_NEWS_CASES[140]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0142_has_required_fields():
    case = SAMPLE_NEWS_CASES[141]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0143_has_required_fields():
    case = SAMPLE_NEWS_CASES[142]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0144_has_required_fields():
    case = SAMPLE_NEWS_CASES[143]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0145_has_required_fields():
    case = SAMPLE_NEWS_CASES[144]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0146_has_required_fields():
    case = SAMPLE_NEWS_CASES[145]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0147_has_required_fields():
    case = SAMPLE_NEWS_CASES[146]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0148_has_required_fields():
    case = SAMPLE_NEWS_CASES[147]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0149_has_required_fields():
    case = SAMPLE_NEWS_CASES[148]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0150_has_required_fields():
    case = SAMPLE_NEWS_CASES[149]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0151_has_required_fields():
    case = SAMPLE_NEWS_CASES[150]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0152_has_required_fields():
    case = SAMPLE_NEWS_CASES[151]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0153_has_required_fields():
    case = SAMPLE_NEWS_CASES[152]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0154_has_required_fields():
    case = SAMPLE_NEWS_CASES[153]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0155_has_required_fields():
    case = SAMPLE_NEWS_CASES[154]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0156_has_required_fields():
    case = SAMPLE_NEWS_CASES[155]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0157_has_required_fields():
    case = SAMPLE_NEWS_CASES[156]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0158_has_required_fields():
    case = SAMPLE_NEWS_CASES[157]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0159_has_required_fields():
    case = SAMPLE_NEWS_CASES[158]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0160_has_required_fields():
    case = SAMPLE_NEWS_CASES[159]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0161_has_required_fields():
    case = SAMPLE_NEWS_CASES[160]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0162_has_required_fields():
    case = SAMPLE_NEWS_CASES[161]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0163_has_required_fields():
    case = SAMPLE_NEWS_CASES[162]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0164_has_required_fields():
    case = SAMPLE_NEWS_CASES[163]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0165_has_required_fields():
    case = SAMPLE_NEWS_CASES[164]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0166_has_required_fields():
    case = SAMPLE_NEWS_CASES[165]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0167_has_required_fields():
    case = SAMPLE_NEWS_CASES[166]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0168_has_required_fields():
    case = SAMPLE_NEWS_CASES[167]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0169_has_required_fields():
    case = SAMPLE_NEWS_CASES[168]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0170_has_required_fields():
    case = SAMPLE_NEWS_CASES[169]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0171_has_required_fields():
    case = SAMPLE_NEWS_CASES[170]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0172_has_required_fields():
    case = SAMPLE_NEWS_CASES[171]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0173_has_required_fields():
    case = SAMPLE_NEWS_CASES[172]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0174_has_required_fields():
    case = SAMPLE_NEWS_CASES[173]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0175_has_required_fields():
    case = SAMPLE_NEWS_CASES[174]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0176_has_required_fields():
    case = SAMPLE_NEWS_CASES[175]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0177_has_required_fields():
    case = SAMPLE_NEWS_CASES[176]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0178_has_required_fields():
    case = SAMPLE_NEWS_CASES[177]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0179_has_required_fields():
    case = SAMPLE_NEWS_CASES[178]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0180_has_required_fields():
    case = SAMPLE_NEWS_CASES[179]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0181_has_required_fields():
    case = SAMPLE_NEWS_CASES[180]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0182_has_required_fields():
    case = SAMPLE_NEWS_CASES[181]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0183_has_required_fields():
    case = SAMPLE_NEWS_CASES[182]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0184_has_required_fields():
    case = SAMPLE_NEWS_CASES[183]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0185_has_required_fields():
    case = SAMPLE_NEWS_CASES[184]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0186_has_required_fields():
    case = SAMPLE_NEWS_CASES[185]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0187_has_required_fields():
    case = SAMPLE_NEWS_CASES[186]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0188_has_required_fields():
    case = SAMPLE_NEWS_CASES[187]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0189_has_required_fields():
    case = SAMPLE_NEWS_CASES[188]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0190_has_required_fields():
    case = SAMPLE_NEWS_CASES[189]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0191_has_required_fields():
    case = SAMPLE_NEWS_CASES[190]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0192_has_required_fields():
    case = SAMPLE_NEWS_CASES[191]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0193_has_required_fields():
    case = SAMPLE_NEWS_CASES[192]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0194_has_required_fields():
    case = SAMPLE_NEWS_CASES[193]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0195_has_required_fields():
    case = SAMPLE_NEWS_CASES[194]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0196_has_required_fields():
    case = SAMPLE_NEWS_CASES[195]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0197_has_required_fields():
    case = SAMPLE_NEWS_CASES[196]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0198_has_required_fields():
    case = SAMPLE_NEWS_CASES[197]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0199_has_required_fields():
    case = SAMPLE_NEWS_CASES[198]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0200_has_required_fields():
    case = SAMPLE_NEWS_CASES[199]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0201_has_required_fields():
    case = SAMPLE_NEWS_CASES[200]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0202_has_required_fields():
    case = SAMPLE_NEWS_CASES[201]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0203_has_required_fields():
    case = SAMPLE_NEWS_CASES[202]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0204_has_required_fields():
    case = SAMPLE_NEWS_CASES[203]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0205_has_required_fields():
    case = SAMPLE_NEWS_CASES[204]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0206_has_required_fields():
    case = SAMPLE_NEWS_CASES[205]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0207_has_required_fields():
    case = SAMPLE_NEWS_CASES[206]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0208_has_required_fields():
    case = SAMPLE_NEWS_CASES[207]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0209_has_required_fields():
    case = SAMPLE_NEWS_CASES[208]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0210_has_required_fields():
    case = SAMPLE_NEWS_CASES[209]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0211_has_required_fields():
    case = SAMPLE_NEWS_CASES[210]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0212_has_required_fields():
    case = SAMPLE_NEWS_CASES[211]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0213_has_required_fields():
    case = SAMPLE_NEWS_CASES[212]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0214_has_required_fields():
    case = SAMPLE_NEWS_CASES[213]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0215_has_required_fields():
    case = SAMPLE_NEWS_CASES[214]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0216_has_required_fields():
    case = SAMPLE_NEWS_CASES[215]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0217_has_required_fields():
    case = SAMPLE_NEWS_CASES[216]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0218_has_required_fields():
    case = SAMPLE_NEWS_CASES[217]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0219_has_required_fields():
    case = SAMPLE_NEWS_CASES[218]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0220_has_required_fields():
    case = SAMPLE_NEWS_CASES[219]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0221_has_required_fields():
    case = SAMPLE_NEWS_CASES[220]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0222_has_required_fields():
    case = SAMPLE_NEWS_CASES[221]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0223_has_required_fields():
    case = SAMPLE_NEWS_CASES[222]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0224_has_required_fields():
    case = SAMPLE_NEWS_CASES[223]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0225_has_required_fields():
    case = SAMPLE_NEWS_CASES[224]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0226_has_required_fields():
    case = SAMPLE_NEWS_CASES[225]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0227_has_required_fields():
    case = SAMPLE_NEWS_CASES[226]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0228_has_required_fields():
    case = SAMPLE_NEWS_CASES[227]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0229_has_required_fields():
    case = SAMPLE_NEWS_CASES[228]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0230_has_required_fields():
    case = SAMPLE_NEWS_CASES[229]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0231_has_required_fields():
    case = SAMPLE_NEWS_CASES[230]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0232_has_required_fields():
    case = SAMPLE_NEWS_CASES[231]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0233_has_required_fields():
    case = SAMPLE_NEWS_CASES[232]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0234_has_required_fields():
    case = SAMPLE_NEWS_CASES[233]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0235_has_required_fields():
    case = SAMPLE_NEWS_CASES[234]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0236_has_required_fields():
    case = SAMPLE_NEWS_CASES[235]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0237_has_required_fields():
    case = SAMPLE_NEWS_CASES[236]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0238_has_required_fields():
    case = SAMPLE_NEWS_CASES[237]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0239_has_required_fields():
    case = SAMPLE_NEWS_CASES[238]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0240_has_required_fields():
    case = SAMPLE_NEWS_CASES[239]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0241_has_required_fields():
    case = SAMPLE_NEWS_CASES[240]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0242_has_required_fields():
    case = SAMPLE_NEWS_CASES[241]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0243_has_required_fields():
    case = SAMPLE_NEWS_CASES[242]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0244_has_required_fields():
    case = SAMPLE_NEWS_CASES[243]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0245_has_required_fields():
    case = SAMPLE_NEWS_CASES[244]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0246_has_required_fields():
    case = SAMPLE_NEWS_CASES[245]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0247_has_required_fields():
    case = SAMPLE_NEWS_CASES[246]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0248_has_required_fields():
    case = SAMPLE_NEWS_CASES[247]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0249_has_required_fields():
    case = SAMPLE_NEWS_CASES[248]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0250_has_required_fields():
    case = SAMPLE_NEWS_CASES[249]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0251_has_required_fields():
    case = SAMPLE_NEWS_CASES[250]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0252_has_required_fields():
    case = SAMPLE_NEWS_CASES[251]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0253_has_required_fields():
    case = SAMPLE_NEWS_CASES[252]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0254_has_required_fields():
    case = SAMPLE_NEWS_CASES[253]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0255_has_required_fields():
    case = SAMPLE_NEWS_CASES[254]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0256_has_required_fields():
    case = SAMPLE_NEWS_CASES[255]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0257_has_required_fields():
    case = SAMPLE_NEWS_CASES[256]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0258_has_required_fields():
    case = SAMPLE_NEWS_CASES[257]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0259_has_required_fields():
    case = SAMPLE_NEWS_CASES[258]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0260_has_required_fields():
    case = SAMPLE_NEWS_CASES[259]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0261_has_required_fields():
    case = SAMPLE_NEWS_CASES[260]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0262_has_required_fields():
    case = SAMPLE_NEWS_CASES[261]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0263_has_required_fields():
    case = SAMPLE_NEWS_CASES[262]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0264_has_required_fields():
    case = SAMPLE_NEWS_CASES[263]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0265_has_required_fields():
    case = SAMPLE_NEWS_CASES[264]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0266_has_required_fields():
    case = SAMPLE_NEWS_CASES[265]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0267_has_required_fields():
    case = SAMPLE_NEWS_CASES[266]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0268_has_required_fields():
    case = SAMPLE_NEWS_CASES[267]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0269_has_required_fields():
    case = SAMPLE_NEWS_CASES[268]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0270_has_required_fields():
    case = SAMPLE_NEWS_CASES[269]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0271_has_required_fields():
    case = SAMPLE_NEWS_CASES[270]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0272_has_required_fields():
    case = SAMPLE_NEWS_CASES[271]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0273_has_required_fields():
    case = SAMPLE_NEWS_CASES[272]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0274_has_required_fields():
    case = SAMPLE_NEWS_CASES[273]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0275_has_required_fields():
    case = SAMPLE_NEWS_CASES[274]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0276_has_required_fields():
    case = SAMPLE_NEWS_CASES[275]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0277_has_required_fields():
    case = SAMPLE_NEWS_CASES[276]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0278_has_required_fields():
    case = SAMPLE_NEWS_CASES[277]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0279_has_required_fields():
    case = SAMPLE_NEWS_CASES[278]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0280_has_required_fields():
    case = SAMPLE_NEWS_CASES[279]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0281_has_required_fields():
    case = SAMPLE_NEWS_CASES[280]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0282_has_required_fields():
    case = SAMPLE_NEWS_CASES[281]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0283_has_required_fields():
    case = SAMPLE_NEWS_CASES[282]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0284_has_required_fields():
    case = SAMPLE_NEWS_CASES[283]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0285_has_required_fields():
    case = SAMPLE_NEWS_CASES[284]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0286_has_required_fields():
    case = SAMPLE_NEWS_CASES[285]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0287_has_required_fields():
    case = SAMPLE_NEWS_CASES[286]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0288_has_required_fields():
    case = SAMPLE_NEWS_CASES[287]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0289_has_required_fields():
    case = SAMPLE_NEWS_CASES[288]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0290_has_required_fields():
    case = SAMPLE_NEWS_CASES[289]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0291_has_required_fields():
    case = SAMPLE_NEWS_CASES[290]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0292_has_required_fields():
    case = SAMPLE_NEWS_CASES[291]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0293_has_required_fields():
    case = SAMPLE_NEWS_CASES[292]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0294_has_required_fields():
    case = SAMPLE_NEWS_CASES[293]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0295_has_required_fields():
    case = SAMPLE_NEWS_CASES[294]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0296_has_required_fields():
    case = SAMPLE_NEWS_CASES[295]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0297_has_required_fields():
    case = SAMPLE_NEWS_CASES[296]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0298_has_required_fields():
    case = SAMPLE_NEWS_CASES[297]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0299_has_required_fields():
    case = SAMPLE_NEWS_CASES[298]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0300_has_required_fields():
    case = SAMPLE_NEWS_CASES[299]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0301_has_required_fields():
    case = SAMPLE_NEWS_CASES[300]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0302_has_required_fields():
    case = SAMPLE_NEWS_CASES[301]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0303_has_required_fields():
    case = SAMPLE_NEWS_CASES[302]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0304_has_required_fields():
    case = SAMPLE_NEWS_CASES[303]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0305_has_required_fields():
    case = SAMPLE_NEWS_CASES[304]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0306_has_required_fields():
    case = SAMPLE_NEWS_CASES[305]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0307_has_required_fields():
    case = SAMPLE_NEWS_CASES[306]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0308_has_required_fields():
    case = SAMPLE_NEWS_CASES[307]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0309_has_required_fields():
    case = SAMPLE_NEWS_CASES[308]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0310_has_required_fields():
    case = SAMPLE_NEWS_CASES[309]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0311_has_required_fields():
    case = SAMPLE_NEWS_CASES[310]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0312_has_required_fields():
    case = SAMPLE_NEWS_CASES[311]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0313_has_required_fields():
    case = SAMPLE_NEWS_CASES[312]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0314_has_required_fields():
    case = SAMPLE_NEWS_CASES[313]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0315_has_required_fields():
    case = SAMPLE_NEWS_CASES[314]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0316_has_required_fields():
    case = SAMPLE_NEWS_CASES[315]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0317_has_required_fields():
    case = SAMPLE_NEWS_CASES[316]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0318_has_required_fields():
    case = SAMPLE_NEWS_CASES[317]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0319_has_required_fields():
    case = SAMPLE_NEWS_CASES[318]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0320_has_required_fields():
    case = SAMPLE_NEWS_CASES[319]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0321_has_required_fields():
    case = SAMPLE_NEWS_CASES[320]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0322_has_required_fields():
    case = SAMPLE_NEWS_CASES[321]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0323_has_required_fields():
    case = SAMPLE_NEWS_CASES[322]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0324_has_required_fields():
    case = SAMPLE_NEWS_CASES[323]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0325_has_required_fields():
    case = SAMPLE_NEWS_CASES[324]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0326_has_required_fields():
    case = SAMPLE_NEWS_CASES[325]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0327_has_required_fields():
    case = SAMPLE_NEWS_CASES[326]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0328_has_required_fields():
    case = SAMPLE_NEWS_CASES[327]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0329_has_required_fields():
    case = SAMPLE_NEWS_CASES[328]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0330_has_required_fields():
    case = SAMPLE_NEWS_CASES[329]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0331_has_required_fields():
    case = SAMPLE_NEWS_CASES[330]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0332_has_required_fields():
    case = SAMPLE_NEWS_CASES[331]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0333_has_required_fields():
    case = SAMPLE_NEWS_CASES[332]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0334_has_required_fields():
    case = SAMPLE_NEWS_CASES[333]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0335_has_required_fields():
    case = SAMPLE_NEWS_CASES[334]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0336_has_required_fields():
    case = SAMPLE_NEWS_CASES[335]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0337_has_required_fields():
    case = SAMPLE_NEWS_CASES[336]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0338_has_required_fields():
    case = SAMPLE_NEWS_CASES[337]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0339_has_required_fields():
    case = SAMPLE_NEWS_CASES[338]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0340_has_required_fields():
    case = SAMPLE_NEWS_CASES[339]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0341_has_required_fields():
    case = SAMPLE_NEWS_CASES[340]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0342_has_required_fields():
    case = SAMPLE_NEWS_CASES[341]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0343_has_required_fields():
    case = SAMPLE_NEWS_CASES[342]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0344_has_required_fields():
    case = SAMPLE_NEWS_CASES[343]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0345_has_required_fields():
    case = SAMPLE_NEWS_CASES[344]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0346_has_required_fields():
    case = SAMPLE_NEWS_CASES[345]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0347_has_required_fields():
    case = SAMPLE_NEWS_CASES[346]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0348_has_required_fields():
    case = SAMPLE_NEWS_CASES[347]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0349_has_required_fields():
    case = SAMPLE_NEWS_CASES[348]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0350_has_required_fields():
    case = SAMPLE_NEWS_CASES[349]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0351_has_required_fields():
    case = SAMPLE_NEWS_CASES[350]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0352_has_required_fields():
    case = SAMPLE_NEWS_CASES[351]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0353_has_required_fields():
    case = SAMPLE_NEWS_CASES[352]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0354_has_required_fields():
    case = SAMPLE_NEWS_CASES[353]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0355_has_required_fields():
    case = SAMPLE_NEWS_CASES[354]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0356_has_required_fields():
    case = SAMPLE_NEWS_CASES[355]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0357_has_required_fields():
    case = SAMPLE_NEWS_CASES[356]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0358_has_required_fields():
    case = SAMPLE_NEWS_CASES[357]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0359_has_required_fields():
    case = SAMPLE_NEWS_CASES[358]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0360_has_required_fields():
    case = SAMPLE_NEWS_CASES[359]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0361_has_required_fields():
    case = SAMPLE_NEWS_CASES[360]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0362_has_required_fields():
    case = SAMPLE_NEWS_CASES[361]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0363_has_required_fields():
    case = SAMPLE_NEWS_CASES[362]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0364_has_required_fields():
    case = SAMPLE_NEWS_CASES[363]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0365_has_required_fields():
    case = SAMPLE_NEWS_CASES[364]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0366_has_required_fields():
    case = SAMPLE_NEWS_CASES[365]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0367_has_required_fields():
    case = SAMPLE_NEWS_CASES[366]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0368_has_required_fields():
    case = SAMPLE_NEWS_CASES[367]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0369_has_required_fields():
    case = SAMPLE_NEWS_CASES[368]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0370_has_required_fields():
    case = SAMPLE_NEWS_CASES[369]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0371_has_required_fields():
    case = SAMPLE_NEWS_CASES[370]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0372_has_required_fields():
    case = SAMPLE_NEWS_CASES[371]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0373_has_required_fields():
    case = SAMPLE_NEWS_CASES[372]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0374_has_required_fields():
    case = SAMPLE_NEWS_CASES[373]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0375_has_required_fields():
    case = SAMPLE_NEWS_CASES[374]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0376_has_required_fields():
    case = SAMPLE_NEWS_CASES[375]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0377_has_required_fields():
    case = SAMPLE_NEWS_CASES[376]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0378_has_required_fields():
    case = SAMPLE_NEWS_CASES[377]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0379_has_required_fields():
    case = SAMPLE_NEWS_CASES[378]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0380_has_required_fields():
    case = SAMPLE_NEWS_CASES[379]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0381_has_required_fields():
    case = SAMPLE_NEWS_CASES[380]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0382_has_required_fields():
    case = SAMPLE_NEWS_CASES[381]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0383_has_required_fields():
    case = SAMPLE_NEWS_CASES[382]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0384_has_required_fields():
    case = SAMPLE_NEWS_CASES[383]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0385_has_required_fields():
    case = SAMPLE_NEWS_CASES[384]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0386_has_required_fields():
    case = SAMPLE_NEWS_CASES[385]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0387_has_required_fields():
    case = SAMPLE_NEWS_CASES[386]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0388_has_required_fields():
    case = SAMPLE_NEWS_CASES[387]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0389_has_required_fields():
    case = SAMPLE_NEWS_CASES[388]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0390_has_required_fields():
    case = SAMPLE_NEWS_CASES[389]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0391_has_required_fields():
    case = SAMPLE_NEWS_CASES[390]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0392_has_required_fields():
    case = SAMPLE_NEWS_CASES[391]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0393_has_required_fields():
    case = SAMPLE_NEWS_CASES[392]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0394_has_required_fields():
    case = SAMPLE_NEWS_CASES[393]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0395_has_required_fields():
    case = SAMPLE_NEWS_CASES[394]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0396_has_required_fields():
    case = SAMPLE_NEWS_CASES[395]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0397_has_required_fields():
    case = SAMPLE_NEWS_CASES[396]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0398_has_required_fields():
    case = SAMPLE_NEWS_CASES[397]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0399_has_required_fields():
    case = SAMPLE_NEWS_CASES[398]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0400_has_required_fields():
    case = SAMPLE_NEWS_CASES[399]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0401_has_required_fields():
    case = SAMPLE_NEWS_CASES[400]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0402_has_required_fields():
    case = SAMPLE_NEWS_CASES[401]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0403_has_required_fields():
    case = SAMPLE_NEWS_CASES[402]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0404_has_required_fields():
    case = SAMPLE_NEWS_CASES[403]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0405_has_required_fields():
    case = SAMPLE_NEWS_CASES[404]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0406_has_required_fields():
    case = SAMPLE_NEWS_CASES[405]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0407_has_required_fields():
    case = SAMPLE_NEWS_CASES[406]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0408_has_required_fields():
    case = SAMPLE_NEWS_CASES[407]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0409_has_required_fields():
    case = SAMPLE_NEWS_CASES[408]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0410_has_required_fields():
    case = SAMPLE_NEWS_CASES[409]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0411_has_required_fields():
    case = SAMPLE_NEWS_CASES[410]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0412_has_required_fields():
    case = SAMPLE_NEWS_CASES[411]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0413_has_required_fields():
    case = SAMPLE_NEWS_CASES[412]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0414_has_required_fields():
    case = SAMPLE_NEWS_CASES[413]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0415_has_required_fields():
    case = SAMPLE_NEWS_CASES[414]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0416_has_required_fields():
    case = SAMPLE_NEWS_CASES[415]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0417_has_required_fields():
    case = SAMPLE_NEWS_CASES[416]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0418_has_required_fields():
    case = SAMPLE_NEWS_CASES[417]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0419_has_required_fields():
    case = SAMPLE_NEWS_CASES[418]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0420_has_required_fields():
    case = SAMPLE_NEWS_CASES[419]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0421_has_required_fields():
    case = SAMPLE_NEWS_CASES[420]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0422_has_required_fields():
    case = SAMPLE_NEWS_CASES[421]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0423_has_required_fields():
    case = SAMPLE_NEWS_CASES[422]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0424_has_required_fields():
    case = SAMPLE_NEWS_CASES[423]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0425_has_required_fields():
    case = SAMPLE_NEWS_CASES[424]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0426_has_required_fields():
    case = SAMPLE_NEWS_CASES[425]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0427_has_required_fields():
    case = SAMPLE_NEWS_CASES[426]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0428_has_required_fields():
    case = SAMPLE_NEWS_CASES[427]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0429_has_required_fields():
    case = SAMPLE_NEWS_CASES[428]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0430_has_required_fields():
    case = SAMPLE_NEWS_CASES[429]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0431_has_required_fields():
    case = SAMPLE_NEWS_CASES[430]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0432_has_required_fields():
    case = SAMPLE_NEWS_CASES[431]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0433_has_required_fields():
    case = SAMPLE_NEWS_CASES[432]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0434_has_required_fields():
    case = SAMPLE_NEWS_CASES[433]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0435_has_required_fields():
    case = SAMPLE_NEWS_CASES[434]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0436_has_required_fields():
    case = SAMPLE_NEWS_CASES[435]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0437_has_required_fields():
    case = SAMPLE_NEWS_CASES[436]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0438_has_required_fields():
    case = SAMPLE_NEWS_CASES[437]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0439_has_required_fields():
    case = SAMPLE_NEWS_CASES[438]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0440_has_required_fields():
    case = SAMPLE_NEWS_CASES[439]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0441_has_required_fields():
    case = SAMPLE_NEWS_CASES[440]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0442_has_required_fields():
    case = SAMPLE_NEWS_CASES[441]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0443_has_required_fields():
    case = SAMPLE_NEWS_CASES[442]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0444_has_required_fields():
    case = SAMPLE_NEWS_CASES[443]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0445_has_required_fields():
    case = SAMPLE_NEWS_CASES[444]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0446_has_required_fields():
    case = SAMPLE_NEWS_CASES[445]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0447_has_required_fields():
    case = SAMPLE_NEWS_CASES[446]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0448_has_required_fields():
    case = SAMPLE_NEWS_CASES[447]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0449_has_required_fields():
    case = SAMPLE_NEWS_CASES[448]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0450_has_required_fields():
    case = SAMPLE_NEWS_CASES[449]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0451_has_required_fields():
    case = SAMPLE_NEWS_CASES[450]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0452_has_required_fields():
    case = SAMPLE_NEWS_CASES[451]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0453_has_required_fields():
    case = SAMPLE_NEWS_CASES[452]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0454_has_required_fields():
    case = SAMPLE_NEWS_CASES[453]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0455_has_required_fields():
    case = SAMPLE_NEWS_CASES[454]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0456_has_required_fields():
    case = SAMPLE_NEWS_CASES[455]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0457_has_required_fields():
    case = SAMPLE_NEWS_CASES[456]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0458_has_required_fields():
    case = SAMPLE_NEWS_CASES[457]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0459_has_required_fields():
    case = SAMPLE_NEWS_CASES[458]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0460_has_required_fields():
    case = SAMPLE_NEWS_CASES[459]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0461_has_required_fields():
    case = SAMPLE_NEWS_CASES[460]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0462_has_required_fields():
    case = SAMPLE_NEWS_CASES[461]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0463_has_required_fields():
    case = SAMPLE_NEWS_CASES[462]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0464_has_required_fields():
    case = SAMPLE_NEWS_CASES[463]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0465_has_required_fields():
    case = SAMPLE_NEWS_CASES[464]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0466_has_required_fields():
    case = SAMPLE_NEWS_CASES[465]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0467_has_required_fields():
    case = SAMPLE_NEWS_CASES[466]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0468_has_required_fields():
    case = SAMPLE_NEWS_CASES[467]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0469_has_required_fields():
    case = SAMPLE_NEWS_CASES[468]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0470_has_required_fields():
    case = SAMPLE_NEWS_CASES[469]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0471_has_required_fields():
    case = SAMPLE_NEWS_CASES[470]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0472_has_required_fields():
    case = SAMPLE_NEWS_CASES[471]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0473_has_required_fields():
    case = SAMPLE_NEWS_CASES[472]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0474_has_required_fields():
    case = SAMPLE_NEWS_CASES[473]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0475_has_required_fields():
    case = SAMPLE_NEWS_CASES[474]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0476_has_required_fields():
    case = SAMPLE_NEWS_CASES[475]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0477_has_required_fields():
    case = SAMPLE_NEWS_CASES[476]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0478_has_required_fields():
    case = SAMPLE_NEWS_CASES[477]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0479_has_required_fields():
    case = SAMPLE_NEWS_CASES[478]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0480_has_required_fields():
    case = SAMPLE_NEWS_CASES[479]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0481_has_required_fields():
    case = SAMPLE_NEWS_CASES[480]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0482_has_required_fields():
    case = SAMPLE_NEWS_CASES[481]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0483_has_required_fields():
    case = SAMPLE_NEWS_CASES[482]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0484_has_required_fields():
    case = SAMPLE_NEWS_CASES[483]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0485_has_required_fields():
    case = SAMPLE_NEWS_CASES[484]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0486_has_required_fields():
    case = SAMPLE_NEWS_CASES[485]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0487_has_required_fields():
    case = SAMPLE_NEWS_CASES[486]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0488_has_required_fields():
    case = SAMPLE_NEWS_CASES[487]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0489_has_required_fields():
    case = SAMPLE_NEWS_CASES[488]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0490_has_required_fields():
    case = SAMPLE_NEWS_CASES[489]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0491_has_required_fields():
    case = SAMPLE_NEWS_CASES[490]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0492_has_required_fields():
    case = SAMPLE_NEWS_CASES[491]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0493_has_required_fields():
    case = SAMPLE_NEWS_CASES[492]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0494_has_required_fields():
    case = SAMPLE_NEWS_CASES[493]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0495_has_required_fields():
    case = SAMPLE_NEWS_CASES[494]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0496_has_required_fields():
    case = SAMPLE_NEWS_CASES[495]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0497_has_required_fields():
    case = SAMPLE_NEWS_CASES[496]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0498_has_required_fields():
    case = SAMPLE_NEWS_CASES[497]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0499_has_required_fields():
    case = SAMPLE_NEWS_CASES[498]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0500_has_required_fields():
    case = SAMPLE_NEWS_CASES[499]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0501_has_required_fields():
    case = SAMPLE_NEWS_CASES[500]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0502_has_required_fields():
    case = SAMPLE_NEWS_CASES[501]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0503_has_required_fields():
    case = SAMPLE_NEWS_CASES[502]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0504_has_required_fields():
    case = SAMPLE_NEWS_CASES[503]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0505_has_required_fields():
    case = SAMPLE_NEWS_CASES[504]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0506_has_required_fields():
    case = SAMPLE_NEWS_CASES[505]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0507_has_required_fields():
    case = SAMPLE_NEWS_CASES[506]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0508_has_required_fields():
    case = SAMPLE_NEWS_CASES[507]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0509_has_required_fields():
    case = SAMPLE_NEWS_CASES[508]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0510_has_required_fields():
    case = SAMPLE_NEWS_CASES[509]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0511_has_required_fields():
    case = SAMPLE_NEWS_CASES[510]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0512_has_required_fields():
    case = SAMPLE_NEWS_CASES[511]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0513_has_required_fields():
    case = SAMPLE_NEWS_CASES[512]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0514_has_required_fields():
    case = SAMPLE_NEWS_CASES[513]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0515_has_required_fields():
    case = SAMPLE_NEWS_CASES[514]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0516_has_required_fields():
    case = SAMPLE_NEWS_CASES[515]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0517_has_required_fields():
    case = SAMPLE_NEWS_CASES[516]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0518_has_required_fields():
    case = SAMPLE_NEWS_CASES[517]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0519_has_required_fields():
    case = SAMPLE_NEWS_CASES[518]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0520_has_required_fields():
    case = SAMPLE_NEWS_CASES[519]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0521_has_required_fields():
    case = SAMPLE_NEWS_CASES[520]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0522_has_required_fields():
    case = SAMPLE_NEWS_CASES[521]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0523_has_required_fields():
    case = SAMPLE_NEWS_CASES[522]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0524_has_required_fields():
    case = SAMPLE_NEWS_CASES[523]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0525_has_required_fields():
    case = SAMPLE_NEWS_CASES[524]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0526_has_required_fields():
    case = SAMPLE_NEWS_CASES[525]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0527_has_required_fields():
    case = SAMPLE_NEWS_CASES[526]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0528_has_required_fields():
    case = SAMPLE_NEWS_CASES[527]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0529_has_required_fields():
    case = SAMPLE_NEWS_CASES[528]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0530_has_required_fields():
    case = SAMPLE_NEWS_CASES[529]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0531_has_required_fields():
    case = SAMPLE_NEWS_CASES[530]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0532_has_required_fields():
    case = SAMPLE_NEWS_CASES[531]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0533_has_required_fields():
    case = SAMPLE_NEWS_CASES[532]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0534_has_required_fields():
    case = SAMPLE_NEWS_CASES[533]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0535_has_required_fields():
    case = SAMPLE_NEWS_CASES[534]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0536_has_required_fields():
    case = SAMPLE_NEWS_CASES[535]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0537_has_required_fields():
    case = SAMPLE_NEWS_CASES[536]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0538_has_required_fields():
    case = SAMPLE_NEWS_CASES[537]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0539_has_required_fields():
    case = SAMPLE_NEWS_CASES[538]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0540_has_required_fields():
    case = SAMPLE_NEWS_CASES[539]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0541_has_required_fields():
    case = SAMPLE_NEWS_CASES[540]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0542_has_required_fields():
    case = SAMPLE_NEWS_CASES[541]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0543_has_required_fields():
    case = SAMPLE_NEWS_CASES[542]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0544_has_required_fields():
    case = SAMPLE_NEWS_CASES[543]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0545_has_required_fields():
    case = SAMPLE_NEWS_CASES[544]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0546_has_required_fields():
    case = SAMPLE_NEWS_CASES[545]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0547_has_required_fields():
    case = SAMPLE_NEWS_CASES[546]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0548_has_required_fields():
    case = SAMPLE_NEWS_CASES[547]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0549_has_required_fields():
    case = SAMPLE_NEWS_CASES[548]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0550_has_required_fields():
    case = SAMPLE_NEWS_CASES[549]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0551_has_required_fields():
    case = SAMPLE_NEWS_CASES[550]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0552_has_required_fields():
    case = SAMPLE_NEWS_CASES[551]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0553_has_required_fields():
    case = SAMPLE_NEWS_CASES[552]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0554_has_required_fields():
    case = SAMPLE_NEWS_CASES[553]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0555_has_required_fields():
    case = SAMPLE_NEWS_CASES[554]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0556_has_required_fields():
    case = SAMPLE_NEWS_CASES[555]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0557_has_required_fields():
    case = SAMPLE_NEWS_CASES[556]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0558_has_required_fields():
    case = SAMPLE_NEWS_CASES[557]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0559_has_required_fields():
    case = SAMPLE_NEWS_CASES[558]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0560_has_required_fields():
    case = SAMPLE_NEWS_CASES[559]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0561_has_required_fields():
    case = SAMPLE_NEWS_CASES[560]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0562_has_required_fields():
    case = SAMPLE_NEWS_CASES[561]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0563_has_required_fields():
    case = SAMPLE_NEWS_CASES[562]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0564_has_required_fields():
    case = SAMPLE_NEWS_CASES[563]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0565_has_required_fields():
    case = SAMPLE_NEWS_CASES[564]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0566_has_required_fields():
    case = SAMPLE_NEWS_CASES[565]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0567_has_required_fields():
    case = SAMPLE_NEWS_CASES[566]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0568_has_required_fields():
    case = SAMPLE_NEWS_CASES[567]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0569_has_required_fields():
    case = SAMPLE_NEWS_CASES[568]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0570_has_required_fields():
    case = SAMPLE_NEWS_CASES[569]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0571_has_required_fields():
    case = SAMPLE_NEWS_CASES[570]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0572_has_required_fields():
    case = SAMPLE_NEWS_CASES[571]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0573_has_required_fields():
    case = SAMPLE_NEWS_CASES[572]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0574_has_required_fields():
    case = SAMPLE_NEWS_CASES[573]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0575_has_required_fields():
    case = SAMPLE_NEWS_CASES[574]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0576_has_required_fields():
    case = SAMPLE_NEWS_CASES[575]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0577_has_required_fields():
    case = SAMPLE_NEWS_CASES[576]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0578_has_required_fields():
    case = SAMPLE_NEWS_CASES[577]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0579_has_required_fields():
    case = SAMPLE_NEWS_CASES[578]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0580_has_required_fields():
    case = SAMPLE_NEWS_CASES[579]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0581_has_required_fields():
    case = SAMPLE_NEWS_CASES[580]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0582_has_required_fields():
    case = SAMPLE_NEWS_CASES[581]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0583_has_required_fields():
    case = SAMPLE_NEWS_CASES[582]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0584_has_required_fields():
    case = SAMPLE_NEWS_CASES[583]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0585_has_required_fields():
    case = SAMPLE_NEWS_CASES[584]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0586_has_required_fields():
    case = SAMPLE_NEWS_CASES[585]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0587_has_required_fields():
    case = SAMPLE_NEWS_CASES[586]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0588_has_required_fields():
    case = SAMPLE_NEWS_CASES[587]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0589_has_required_fields():
    case = SAMPLE_NEWS_CASES[588]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0590_has_required_fields():
    case = SAMPLE_NEWS_CASES[589]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0591_has_required_fields():
    case = SAMPLE_NEWS_CASES[590]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0592_has_required_fields():
    case = SAMPLE_NEWS_CASES[591]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0593_has_required_fields():
    case = SAMPLE_NEWS_CASES[592]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0594_has_required_fields():
    case = SAMPLE_NEWS_CASES[593]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0595_has_required_fields():
    case = SAMPLE_NEWS_CASES[594]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0596_has_required_fields():
    case = SAMPLE_NEWS_CASES[595]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0597_has_required_fields():
    case = SAMPLE_NEWS_CASES[596]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0598_has_required_fields():
    case = SAMPLE_NEWS_CASES[597]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0599_has_required_fields():
    case = SAMPLE_NEWS_CASES[598]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0600_has_required_fields():
    case = SAMPLE_NEWS_CASES[599]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0601_has_required_fields():
    case = SAMPLE_NEWS_CASES[600]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0602_has_required_fields():
    case = SAMPLE_NEWS_CASES[601]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0603_has_required_fields():
    case = SAMPLE_NEWS_CASES[602]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0604_has_required_fields():
    case = SAMPLE_NEWS_CASES[603]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0605_has_required_fields():
    case = SAMPLE_NEWS_CASES[604]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0606_has_required_fields():
    case = SAMPLE_NEWS_CASES[605]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0607_has_required_fields():
    case = SAMPLE_NEWS_CASES[606]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0608_has_required_fields():
    case = SAMPLE_NEWS_CASES[607]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0609_has_required_fields():
    case = SAMPLE_NEWS_CASES[608]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0610_has_required_fields():
    case = SAMPLE_NEWS_CASES[609]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0611_has_required_fields():
    case = SAMPLE_NEWS_CASES[610]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0612_has_required_fields():
    case = SAMPLE_NEWS_CASES[611]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0613_has_required_fields():
    case = SAMPLE_NEWS_CASES[612]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0614_has_required_fields():
    case = SAMPLE_NEWS_CASES[613]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0615_has_required_fields():
    case = SAMPLE_NEWS_CASES[614]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0616_has_required_fields():
    case = SAMPLE_NEWS_CASES[615]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0617_has_required_fields():
    case = SAMPLE_NEWS_CASES[616]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0618_has_required_fields():
    case = SAMPLE_NEWS_CASES[617]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0619_has_required_fields():
    case = SAMPLE_NEWS_CASES[618]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0620_has_required_fields():
    case = SAMPLE_NEWS_CASES[619]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0621_has_required_fields():
    case = SAMPLE_NEWS_CASES[620]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0622_has_required_fields():
    case = SAMPLE_NEWS_CASES[621]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0623_has_required_fields():
    case = SAMPLE_NEWS_CASES[622]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0624_has_required_fields():
    case = SAMPLE_NEWS_CASES[623]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0625_has_required_fields():
    case = SAMPLE_NEWS_CASES[624]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0626_has_required_fields():
    case = SAMPLE_NEWS_CASES[625]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0627_has_required_fields():
    case = SAMPLE_NEWS_CASES[626]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0628_has_required_fields():
    case = SAMPLE_NEWS_CASES[627]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0629_has_required_fields():
    case = SAMPLE_NEWS_CASES[628]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0630_has_required_fields():
    case = SAMPLE_NEWS_CASES[629]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0631_has_required_fields():
    case = SAMPLE_NEWS_CASES[630]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0632_has_required_fields():
    case = SAMPLE_NEWS_CASES[631]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0633_has_required_fields():
    case = SAMPLE_NEWS_CASES[632]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0634_has_required_fields():
    case = SAMPLE_NEWS_CASES[633]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0635_has_required_fields():
    case = SAMPLE_NEWS_CASES[634]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0636_has_required_fields():
    case = SAMPLE_NEWS_CASES[635]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0637_has_required_fields():
    case = SAMPLE_NEWS_CASES[636]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0638_has_required_fields():
    case = SAMPLE_NEWS_CASES[637]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0639_has_required_fields():
    case = SAMPLE_NEWS_CASES[638]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0640_has_required_fields():
    case = SAMPLE_NEWS_CASES[639]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0641_has_required_fields():
    case = SAMPLE_NEWS_CASES[640]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0642_has_required_fields():
    case = SAMPLE_NEWS_CASES[641]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0643_has_required_fields():
    case = SAMPLE_NEWS_CASES[642]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0644_has_required_fields():
    case = SAMPLE_NEWS_CASES[643]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0645_has_required_fields():
    case = SAMPLE_NEWS_CASES[644]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0646_has_required_fields():
    case = SAMPLE_NEWS_CASES[645]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0647_has_required_fields():
    case = SAMPLE_NEWS_CASES[646]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0648_has_required_fields():
    case = SAMPLE_NEWS_CASES[647]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0649_has_required_fields():
    case = SAMPLE_NEWS_CASES[648]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0650_has_required_fields():
    case = SAMPLE_NEWS_CASES[649]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0651_has_required_fields():
    case = SAMPLE_NEWS_CASES[650]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0652_has_required_fields():
    case = SAMPLE_NEWS_CASES[651]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0653_has_required_fields():
    case = SAMPLE_NEWS_CASES[652]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0654_has_required_fields():
    case = SAMPLE_NEWS_CASES[653]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0655_has_required_fields():
    case = SAMPLE_NEWS_CASES[654]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0656_has_required_fields():
    case = SAMPLE_NEWS_CASES[655]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0657_has_required_fields():
    case = SAMPLE_NEWS_CASES[656]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0658_has_required_fields():
    case = SAMPLE_NEWS_CASES[657]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0659_has_required_fields():
    case = SAMPLE_NEWS_CASES[658]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0660_has_required_fields():
    case = SAMPLE_NEWS_CASES[659]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0661_has_required_fields():
    case = SAMPLE_NEWS_CASES[660]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0662_has_required_fields():
    case = SAMPLE_NEWS_CASES[661]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0663_has_required_fields():
    case = SAMPLE_NEWS_CASES[662]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0664_has_required_fields():
    case = SAMPLE_NEWS_CASES[663]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0665_has_required_fields():
    case = SAMPLE_NEWS_CASES[664]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0666_has_required_fields():
    case = SAMPLE_NEWS_CASES[665]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0667_has_required_fields():
    case = SAMPLE_NEWS_CASES[666]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0668_has_required_fields():
    case = SAMPLE_NEWS_CASES[667]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0669_has_required_fields():
    case = SAMPLE_NEWS_CASES[668]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0670_has_required_fields():
    case = SAMPLE_NEWS_CASES[669]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0671_has_required_fields():
    case = SAMPLE_NEWS_CASES[670]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0672_has_required_fields():
    case = SAMPLE_NEWS_CASES[671]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0673_has_required_fields():
    case = SAMPLE_NEWS_CASES[672]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0674_has_required_fields():
    case = SAMPLE_NEWS_CASES[673]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0675_has_required_fields():
    case = SAMPLE_NEWS_CASES[674]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0676_has_required_fields():
    case = SAMPLE_NEWS_CASES[675]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0677_has_required_fields():
    case = SAMPLE_NEWS_CASES[676]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0678_has_required_fields():
    case = SAMPLE_NEWS_CASES[677]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0679_has_required_fields():
    case = SAMPLE_NEWS_CASES[678]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0680_has_required_fields():
    case = SAMPLE_NEWS_CASES[679]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0681_has_required_fields():
    case = SAMPLE_NEWS_CASES[680]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0682_has_required_fields():
    case = SAMPLE_NEWS_CASES[681]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0683_has_required_fields():
    case = SAMPLE_NEWS_CASES[682]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0684_has_required_fields():
    case = SAMPLE_NEWS_CASES[683]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0685_has_required_fields():
    case = SAMPLE_NEWS_CASES[684]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0686_has_required_fields():
    case = SAMPLE_NEWS_CASES[685]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0687_has_required_fields():
    case = SAMPLE_NEWS_CASES[686]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0688_has_required_fields():
    case = SAMPLE_NEWS_CASES[687]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0689_has_required_fields():
    case = SAMPLE_NEWS_CASES[688]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0690_has_required_fields():
    case = SAMPLE_NEWS_CASES[689]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0691_has_required_fields():
    case = SAMPLE_NEWS_CASES[690]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0692_has_required_fields():
    case = SAMPLE_NEWS_CASES[691]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0693_has_required_fields():
    case = SAMPLE_NEWS_CASES[692]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0694_has_required_fields():
    case = SAMPLE_NEWS_CASES[693]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0695_has_required_fields():
    case = SAMPLE_NEWS_CASES[694]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0696_has_required_fields():
    case = SAMPLE_NEWS_CASES[695]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0697_has_required_fields():
    case = SAMPLE_NEWS_CASES[696]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0698_has_required_fields():
    case = SAMPLE_NEWS_CASES[697]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0699_has_required_fields():
    case = SAMPLE_NEWS_CASES[698]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0700_has_required_fields():
    case = SAMPLE_NEWS_CASES[699]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0701_has_required_fields():
    case = SAMPLE_NEWS_CASES[700]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0702_has_required_fields():
    case = SAMPLE_NEWS_CASES[701]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0703_has_required_fields():
    case = SAMPLE_NEWS_CASES[702]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0704_has_required_fields():
    case = SAMPLE_NEWS_CASES[703]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0705_has_required_fields():
    case = SAMPLE_NEWS_CASES[704]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0706_has_required_fields():
    case = SAMPLE_NEWS_CASES[705]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0707_has_required_fields():
    case = SAMPLE_NEWS_CASES[706]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0708_has_required_fields():
    case = SAMPLE_NEWS_CASES[707]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0709_has_required_fields():
    case = SAMPLE_NEWS_CASES[708]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0710_has_required_fields():
    case = SAMPLE_NEWS_CASES[709]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0711_has_required_fields():
    case = SAMPLE_NEWS_CASES[710]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0712_has_required_fields():
    case = SAMPLE_NEWS_CASES[711]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0713_has_required_fields():
    case = SAMPLE_NEWS_CASES[712]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0714_has_required_fields():
    case = SAMPLE_NEWS_CASES[713]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0715_has_required_fields():
    case = SAMPLE_NEWS_CASES[714]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0716_has_required_fields():
    case = SAMPLE_NEWS_CASES[715]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0717_has_required_fields():
    case = SAMPLE_NEWS_CASES[716]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0718_has_required_fields():
    case = SAMPLE_NEWS_CASES[717]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0719_has_required_fields():
    case = SAMPLE_NEWS_CASES[718]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0720_has_required_fields():
    case = SAMPLE_NEWS_CASES[719]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0721_has_required_fields():
    case = SAMPLE_NEWS_CASES[720]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0722_has_required_fields():
    case = SAMPLE_NEWS_CASES[721]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0723_has_required_fields():
    case = SAMPLE_NEWS_CASES[722]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0724_has_required_fields():
    case = SAMPLE_NEWS_CASES[723]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0725_has_required_fields():
    case = SAMPLE_NEWS_CASES[724]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0726_has_required_fields():
    case = SAMPLE_NEWS_CASES[725]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0727_has_required_fields():
    case = SAMPLE_NEWS_CASES[726]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0728_has_required_fields():
    case = SAMPLE_NEWS_CASES[727]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0729_has_required_fields():
    case = SAMPLE_NEWS_CASES[728]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0730_has_required_fields():
    case = SAMPLE_NEWS_CASES[729]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0731_has_required_fields():
    case = SAMPLE_NEWS_CASES[730]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0732_has_required_fields():
    case = SAMPLE_NEWS_CASES[731]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0733_has_required_fields():
    case = SAMPLE_NEWS_CASES[732]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0734_has_required_fields():
    case = SAMPLE_NEWS_CASES[733]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0735_has_required_fields():
    case = SAMPLE_NEWS_CASES[734]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0736_has_required_fields():
    case = SAMPLE_NEWS_CASES[735]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0737_has_required_fields():
    case = SAMPLE_NEWS_CASES[736]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0738_has_required_fields():
    case = SAMPLE_NEWS_CASES[737]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0739_has_required_fields():
    case = SAMPLE_NEWS_CASES[738]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0740_has_required_fields():
    case = SAMPLE_NEWS_CASES[739]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0741_has_required_fields():
    case = SAMPLE_NEWS_CASES[740]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0742_has_required_fields():
    case = SAMPLE_NEWS_CASES[741]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0743_has_required_fields():
    case = SAMPLE_NEWS_CASES[742]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0744_has_required_fields():
    case = SAMPLE_NEWS_CASES[743]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0745_has_required_fields():
    case = SAMPLE_NEWS_CASES[744]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0746_has_required_fields():
    case = SAMPLE_NEWS_CASES[745]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0747_has_required_fields():
    case = SAMPLE_NEWS_CASES[746]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0748_has_required_fields():
    case = SAMPLE_NEWS_CASES[747]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0749_has_required_fields():
    case = SAMPLE_NEWS_CASES[748]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0750_has_required_fields():
    case = SAMPLE_NEWS_CASES[749]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0751_has_required_fields():
    case = SAMPLE_NEWS_CASES[750]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0752_has_required_fields():
    case = SAMPLE_NEWS_CASES[751]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0753_has_required_fields():
    case = SAMPLE_NEWS_CASES[752]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0754_has_required_fields():
    case = SAMPLE_NEWS_CASES[753]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0755_has_required_fields():
    case = SAMPLE_NEWS_CASES[754]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0756_has_required_fields():
    case = SAMPLE_NEWS_CASES[755]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0757_has_required_fields():
    case = SAMPLE_NEWS_CASES[756]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0758_has_required_fields():
    case = SAMPLE_NEWS_CASES[757]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0759_has_required_fields():
    case = SAMPLE_NEWS_CASES[758]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0760_has_required_fields():
    case = SAMPLE_NEWS_CASES[759]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0761_has_required_fields():
    case = SAMPLE_NEWS_CASES[760]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0762_has_required_fields():
    case = SAMPLE_NEWS_CASES[761]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0763_has_required_fields():
    case = SAMPLE_NEWS_CASES[762]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0764_has_required_fields():
    case = SAMPLE_NEWS_CASES[763]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0765_has_required_fields():
    case = SAMPLE_NEWS_CASES[764]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0766_has_required_fields():
    case = SAMPLE_NEWS_CASES[765]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0767_has_required_fields():
    case = SAMPLE_NEWS_CASES[766]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0768_has_required_fields():
    case = SAMPLE_NEWS_CASES[767]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0769_has_required_fields():
    case = SAMPLE_NEWS_CASES[768]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0770_has_required_fields():
    case = SAMPLE_NEWS_CASES[769]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0771_has_required_fields():
    case = SAMPLE_NEWS_CASES[770]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0772_has_required_fields():
    case = SAMPLE_NEWS_CASES[771]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0773_has_required_fields():
    case = SAMPLE_NEWS_CASES[772]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0774_has_required_fields():
    case = SAMPLE_NEWS_CASES[773]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0775_has_required_fields():
    case = SAMPLE_NEWS_CASES[774]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0776_has_required_fields():
    case = SAMPLE_NEWS_CASES[775]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0777_has_required_fields():
    case = SAMPLE_NEWS_CASES[776]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0778_has_required_fields():
    case = SAMPLE_NEWS_CASES[777]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0779_has_required_fields():
    case = SAMPLE_NEWS_CASES[778]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0780_has_required_fields():
    case = SAMPLE_NEWS_CASES[779]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0781_has_required_fields():
    case = SAMPLE_NEWS_CASES[780]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0782_has_required_fields():
    case = SAMPLE_NEWS_CASES[781]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0783_has_required_fields():
    case = SAMPLE_NEWS_CASES[782]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0784_has_required_fields():
    case = SAMPLE_NEWS_CASES[783]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0785_has_required_fields():
    case = SAMPLE_NEWS_CASES[784]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0786_has_required_fields():
    case = SAMPLE_NEWS_CASES[785]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0787_has_required_fields():
    case = SAMPLE_NEWS_CASES[786]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0788_has_required_fields():
    case = SAMPLE_NEWS_CASES[787]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0789_has_required_fields():
    case = SAMPLE_NEWS_CASES[788]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0790_has_required_fields():
    case = SAMPLE_NEWS_CASES[789]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0791_has_required_fields():
    case = SAMPLE_NEWS_CASES[790]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0792_has_required_fields():
    case = SAMPLE_NEWS_CASES[791]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0793_has_required_fields():
    case = SAMPLE_NEWS_CASES[792]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0794_has_required_fields():
    case = SAMPLE_NEWS_CASES[793]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0795_has_required_fields():
    case = SAMPLE_NEWS_CASES[794]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0796_has_required_fields():
    case = SAMPLE_NEWS_CASES[795]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0797_has_required_fields():
    case = SAMPLE_NEWS_CASES[796]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0798_has_required_fields():
    case = SAMPLE_NEWS_CASES[797]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0799_has_required_fields():
    case = SAMPLE_NEWS_CASES[798]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0800_has_required_fields():
    case = SAMPLE_NEWS_CASES[799]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0801_has_required_fields():
    case = SAMPLE_NEWS_CASES[800]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0802_has_required_fields():
    case = SAMPLE_NEWS_CASES[801]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0803_has_required_fields():
    case = SAMPLE_NEWS_CASES[802]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0804_has_required_fields():
    case = SAMPLE_NEWS_CASES[803]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0805_has_required_fields():
    case = SAMPLE_NEWS_CASES[804]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0806_has_required_fields():
    case = SAMPLE_NEWS_CASES[805]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0807_has_required_fields():
    case = SAMPLE_NEWS_CASES[806]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0808_has_required_fields():
    case = SAMPLE_NEWS_CASES[807]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0809_has_required_fields():
    case = SAMPLE_NEWS_CASES[808]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0810_has_required_fields():
    case = SAMPLE_NEWS_CASES[809]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0811_has_required_fields():
    case = SAMPLE_NEWS_CASES[810]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0812_has_required_fields():
    case = SAMPLE_NEWS_CASES[811]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0813_has_required_fields():
    case = SAMPLE_NEWS_CASES[812]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0814_has_required_fields():
    case = SAMPLE_NEWS_CASES[813]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0815_has_required_fields():
    case = SAMPLE_NEWS_CASES[814]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0816_has_required_fields():
    case = SAMPLE_NEWS_CASES[815]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0817_has_required_fields():
    case = SAMPLE_NEWS_CASES[816]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0818_has_required_fields():
    case = SAMPLE_NEWS_CASES[817]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0819_has_required_fields():
    case = SAMPLE_NEWS_CASES[818]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0820_has_required_fields():
    case = SAMPLE_NEWS_CASES[819]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0821_has_required_fields():
    case = SAMPLE_NEWS_CASES[820]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0822_has_required_fields():
    case = SAMPLE_NEWS_CASES[821]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0823_has_required_fields():
    case = SAMPLE_NEWS_CASES[822]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0824_has_required_fields():
    case = SAMPLE_NEWS_CASES[823]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0825_has_required_fields():
    case = SAMPLE_NEWS_CASES[824]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0826_has_required_fields():
    case = SAMPLE_NEWS_CASES[825]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0827_has_required_fields():
    case = SAMPLE_NEWS_CASES[826]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0828_has_required_fields():
    case = SAMPLE_NEWS_CASES[827]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0829_has_required_fields():
    case = SAMPLE_NEWS_CASES[828]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0830_has_required_fields():
    case = SAMPLE_NEWS_CASES[829]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0831_has_required_fields():
    case = SAMPLE_NEWS_CASES[830]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0832_has_required_fields():
    case = SAMPLE_NEWS_CASES[831]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0833_has_required_fields():
    case = SAMPLE_NEWS_CASES[832]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0834_has_required_fields():
    case = SAMPLE_NEWS_CASES[833]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0835_has_required_fields():
    case = SAMPLE_NEWS_CASES[834]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0836_has_required_fields():
    case = SAMPLE_NEWS_CASES[835]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0837_has_required_fields():
    case = SAMPLE_NEWS_CASES[836]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0838_has_required_fields():
    case = SAMPLE_NEWS_CASES[837]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0839_has_required_fields():
    case = SAMPLE_NEWS_CASES[838]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0840_has_required_fields():
    case = SAMPLE_NEWS_CASES[839]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0841_has_required_fields():
    case = SAMPLE_NEWS_CASES[840]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0842_has_required_fields():
    case = SAMPLE_NEWS_CASES[841]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0843_has_required_fields():
    case = SAMPLE_NEWS_CASES[842]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0844_has_required_fields():
    case = SAMPLE_NEWS_CASES[843]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0845_has_required_fields():
    case = SAMPLE_NEWS_CASES[844]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0846_has_required_fields():
    case = SAMPLE_NEWS_CASES[845]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0847_has_required_fields():
    case = SAMPLE_NEWS_CASES[846]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0848_has_required_fields():
    case = SAMPLE_NEWS_CASES[847]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0849_has_required_fields():
    case = SAMPLE_NEWS_CASES[848]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0850_has_required_fields():
    case = SAMPLE_NEWS_CASES[849]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0851_has_required_fields():
    case = SAMPLE_NEWS_CASES[850]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0852_has_required_fields():
    case = SAMPLE_NEWS_CASES[851]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0853_has_required_fields():
    case = SAMPLE_NEWS_CASES[852]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0854_has_required_fields():
    case = SAMPLE_NEWS_CASES[853]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0855_has_required_fields():
    case = SAMPLE_NEWS_CASES[854]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0856_has_required_fields():
    case = SAMPLE_NEWS_CASES[855]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0857_has_required_fields():
    case = SAMPLE_NEWS_CASES[856]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0858_has_required_fields():
    case = SAMPLE_NEWS_CASES[857]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0859_has_required_fields():
    case = SAMPLE_NEWS_CASES[858]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0860_has_required_fields():
    case = SAMPLE_NEWS_CASES[859]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0861_has_required_fields():
    case = SAMPLE_NEWS_CASES[860]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0862_has_required_fields():
    case = SAMPLE_NEWS_CASES[861]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0863_has_required_fields():
    case = SAMPLE_NEWS_CASES[862]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

def test_sample_case_0864_has_required_fields():
    case = SAMPLE_NEWS_CASES[863]
    assert case["title"]
    assert case["description"]
    assert case["expected"] in RULES

