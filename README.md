# 뉴소트(NewSort)

검색어로 뉴스를 자동 수집하고, 다국어 AI와 보정 규칙으로 경제, 산업과 일반 뉴스의 17개 주제에 분류하는 Python 기반 뉴스 분석 프로젝트.
분류 결과와 판단 근거의 CSV 저장, 대시보드 시각화, 터미널에서 실행 성공과 오류 구분.

## 핵심 기능

- **수집과 정제:** Google News RSS의 최근 7일 뉴스 수집, 제목 정리, 완전 중복 및 동일 사건의 유사 기사 묶기
- **AI 분류:** 사전학습 모델 mDeBERTa와 사건 중심 주제 설명을 이용한 별도 추가 학습 없는 분류
- **규칙 기반 판단:** 모델 예측 점수와 상위 두 주제의 점수 차이, 기사 속 키워드 근거 비교를 통한 최종 판단
- **검토 기사 재판단:** Google 뉴스 링크의 언론사 원문 조회, 본문 보강과 최대 1회 재분류, 최초 판단과 처리 결과 기록
- **규칙 오탐 개선:** 단어 경계 확인, 긴 구문 우선, 중복 가산 방지, 분야명과 고유명사의 점수 제외, 강한 근거 2점과 일반 근거 1점 적용
- **결과 관리:** CSV 저장과 Streamlit 대시보드 조회, 선택적 SQLite 누적 저장

분류 주제: 기술개발, 제품/서비스, 기업동향, 생산/공급망, 정책/규제, 금융/투자, 시장/산업, 노동/노사, 국제/통상, 교육/취업, 사회, 정치, 문화/연예, 스포츠, 건강/의료, 생활/환경, 기타/무관.

새 일반 뉴스 주제는 현재 직접 규칙 없이 모델 판단으로 분류. 서로 다른 사건의 사람이 확인한 오류 근거 확보 후에만 규칙 활성화 가능, 임의의 새 키워드 규칙 추가 없음.
정책/규제는 정부 정책과 법률, 정치는 정당과 선거 중심 구분. 교육/취업은 입시와 교육, 직업 훈련과 구직, 노동/노사는 임금과 근로 조건 및 노사 갈등 중심 구분.

사용 기술: Python, PyTorch, Transformers, Pandas, BeautifulSoup, Streamlit, SQLite, Pytest.

## 처리 흐름

검색어 입력 → RSS 수집 → 정제와 중복 제거 → AI 분류 → 규칙 기반 최종 판단 → CSV 저장과 결과 확인.
본문이 없는 검토 기사만 기본 본문 보강 후 재판단, 정보 확보 실패 또는 재판단 후 불확실할 경우 검토 유지. 전체 기사 본문 보강과 SQLite 저장은 선택 기능.

![뉴스 수집부터 분류와 저장까지의 아키텍처](docs/architecture.png)

## 실행 방법

실행 환경: Python 3.11~3.13, 뉴스 수집과 최초 모델 다운로드를 위한 인터넷 연결.
아래 명령은 Windows PowerShell 기준이며, 최초 분류 시 모델 다운로드와 로딩 시간 소요 가능.

### 1. 프로젝트 폴더 이동

작업 위치: `pyproject.toml`이 있는 `news_classifier_expanded` 폴더.
상위 폴더에서 시작한 경우 아래 명령으로 이동, 이미 해당 폴더인 경우 생략.

```powershell
cd .\news_classifier_expanded
```

### 2. 상위 가상환경 활성화

가상환경: 다른 Python 프로젝트와 사용 패키지를 분리하는 실행 공간.
이 작업 환경은 `C:\news_classifier_expanded_project\.venv` 하나만 사용. 프로젝트 내부 `news_classifier_expanded\.venv`는 생성하지 않음. 상위 환경의 GPU 지원 PyTorch와 프로젝트 설치 확인 완료, 기존 환경은 재생성이나 재설치 불필요.

```powershell
# 기존 터미널에서 다른 가상환경이 활성화되어 있으면 먼저 deactivate 실행
..\.venv\Scripts\Activate.ps1
```

새 컴퓨터에서 상위 가상환경이 없는 경우에만 아래 명령으로 준비. GPU 사용에는 해당 환경의 CUDA 지원 PyTorch 및 GPU 사용 가능 여부 확인 필요.

```powershell
python -m venv ..\.venv
..\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

macOS와 Linux에서는 활성화 명령을 `source ../.venv/bin/activate`로 대체.
기본 실행은 `.env` 파일 생성 없이 가능, 설정 변경 시 `.env.example`을 복사한 `.env` 파일 사용.

### 3. 뉴스 수집과 분류

```powershell
news-classifier collect --keyword "AI 반도체"
```

기본 최대 수집량 10건, 정제와 동일 사건 묶기 후 대표 기사 수 감소 가능. 공통 단어만 겹치는 서로 다른 기사는 유지.
가상환경을 활성화하지 않고 상위 Python을 직접 사용하려면 `..\.venv\Scripts\python.exe -m news_classifier.cli collect --keyword "실제 검색어"` 실행. `run_news.ps1`도 상위 Python으로 고정, `./run_news.ps1 collect --keyword "실제 검색어"` 또는 `./run_news.ps1 dashboard` 사용 가능. 인수 없이 실행하면 기존의 대화형 검색과 `out.csv`, `news.db` 저장 방식 유지.

### 4. 대시보드 확인

```powershell
news-classifier dashboard
```

CSV 결과의 주제별 기사 수, 규칙 적용 수, 검토 필요 기사 확인.
그래프는 최초 모델 예측인 `model_category`가 아닌 최종 판단인 `final_category` 기준 집계, 실제 결과에 있는 주제만 표시.
수집 기사 수와 대표 기사 수 구분, `묶인 기사와 원문 링크 보기`에서 다른 언론사의 기사 확인.
대시보드 종료는 실행한 터미널에서 `Ctrl+C` 입력.

### 선택 옵션

| 옵션 | 용도 |
|---|---|
| `--limit 20` | 최대 수집량 지정, 허용 범위 1~100 |
| `--csv result.csv` | CSV 저장 경로 지정 |
| `--sqlite` | SQLite에도 저장, 기본 파일 `news_analysis.db` |
| `--sqlite custom.db` | SQLite 저장 경로 직접 지정 |
| `--enrich-content` | 최초 분류 전 전체 기사 본문 보강 시도, 같은 실행의 검토 기사 중복 보강 제외 |
| `--no-review-enrichment` | 기본 검토 기사 본문 보강과 재판단 비활성화 |
| `--confidence-profile FILE.json` | 모델 신뢰도 높음/보통/낮음 표시 기준 지정, 최종 판정 기준과 별개 |
| `--decision-profile FILE.json` | 독립 평가를 통과한 최종 판정 기준 적용, 입력 조건 불일치 시 기본 기준 유지 |
| `--rule-profile FILE.json` | 사람이 확인한 기사로 독립 평가를 통과한 사건 규칙 적용, 후보 감사 파일 직접 사용 불가 |

```powershell
news-classifier collect --keyword "AI 반도체" --limit 20 --sqlite
```

## 실행 예시

검색어 입력과 수집 명령 예시.

![뉴스 수집과 분류 명령 입력 화면](image.png)

기사별 분류 결과, 실행 상태, CSV 저장 경로와 처리 건수의 터미널 출력 예시.
검색 시점과 기사 내용에 따른 결과 변동 가능. 아래 사진은 이전 주제 체계의 실행 예시이며, 최신 17개 주제 결과는 재수집 후 확인.

![뉴스 분류 결과와 실행 상태 출력 화면](image-1.png)

## 결과 읽는 방법

- **CSV:** 기본 파일 `news_analysis_results.csv`, Excel에서 열기 가능, 실행마다 해당 파일의 결과 교체
- **SQLite:** 옵션 지정 시 결과 누적 보관, 같은 검색어와 제목 및 언론사의 기사는 기존 항목 갱신
- **대시보드:** CSV에 저장된 기사별 결과 조회용 화면, 전체 실행 성공과 실패는 터미널에서 확인

`decision_source`는 최종 판단의 출처이며, 정상 실행에서 사용하는 판정 경로는 아래 3종.

| 값 | 의미 |
|---|---|
| `MODEL` | 모델 예측 유지 |
| `RULE` | 규칙 근거로 최종 주제 결정 |
| `REVIEW` | 모델 판단도 불확실하고 규칙 근거도 부족하여 사람 검토 필요 |

`final_category`는 최종 주제 또는 검토필요 상태, `rule_reason`은 보정이나 검토 판단의 이유.
검토 사유는 `근거 없음`, `규칙 점수 부족`, `주제 간 점수 차이 부족`으로 구분. 여러 조건이 부족한 경우 근거 없음, 총점 부족, 점수 차이 부족 순서로 우선 표시.
새 개발 규칙이 실제 점수에 기여하고 여러 주제의 강한 근거가 충돌하면 `주제 근거 충돌`로 별도 기록. 불확실한 모델의 규칙 강제 보정 방지, 기존 규칙만의 판단과 명확한 모델 유지 보존.
`model_confidence`는 원래 모델 예측의 참고 점수이며, 규칙 적용 후 최종 주제의 정답 확률을 보장하는 값은 아님.
검증된 판정 프로필로 기본 기준의 검토 기사를 새로 자동 분류하면 `rule_reason`에 적용 점수 기준과 기존 검토 사유 기록. 이 경우에도 `decision_source=MODEL`과 모델 원점수 유지, 신뢰도 표시가 낮음이어도 검증된 기준에 따른 자동 분류 가능.

`group_article_count`는 대표를 포함한 묶음 기사 수, `related_articles`는 다른 기사들의 제목, 언론사, 발행일과 원문 링크를 보존한 JSON 목록. 분류 입력에는 대표 기사의 내용만 사용.
`review_reclassification`은 최초 모델 점수와 검토 사유, 본문 보강과 재판단 상태의 JSON 기록. `RECLASSIFIED`는 재판단 수행이며 정답 확정이나 검토 해소를 보장하는 상태는 아님. `enrichment_status`에 본문 확보, 시간 초과, 접근 차단, Google 뉴스 중계 페이지, RSS 문서, 본문 문단 없음 등의 원인 기록. HTTP 응답이 있으면 `http_status`도 기록. 이전 CSV의 `NO_CONTENT`는 상세 원인 없는 과거 기록이며 자동 변경 없음. 모델 재판단 실패는 전체 실행 오류로 처리하고 저장 중단.
`resolved_url`은 조회한 언론사 URL, `resolution_status`는 조회 경로 기록. 원래 기사 링크와 중복 묶음 정보 보존. 원문 주소 조회 실패는 `SOURCE_URL_UNRESOLVED`, 허용하지 않는 주소는 `UNSAFE_URL`, 리다이렉트 한도 초과는 `REDIRECT_LIMIT`으로 구분.
`extraction_diagnostics`는 본문 선택 방식, 실패 이유, 문단 수와 후보 영역 수 기록. 첫 번째 `<article>` 대신 본문 전용 영역 우선 선택, `<div>`와 줄바꿈 본문 지원. 기자 소개와 메뉴, 추천 기사 및 광고 제외. 여러 독립 본문이 충돌하면 추측 없이 검토 유지.
터미널과 대시보드의 카테고리 통계 및 검토 수는 대표 기사 기준. 대표는 수집 순서상 첫 기사이며, 모델 점수에 따른 대표 선택 없음.

- **정상 분류 또는 검색 결과 0건:** 종료 코드 `0`, 0건이면 열 이름(헤더)만 있는 CSV 저장
- **수집 또는 모델 오류:** 오류 메시지 출력, 실패한 실행 결과의 저장 중단, 종료 코드 `1`
- **입력 기사 수와 모델 출력 수 불일치:** 모델 오류 처리로 기사 누락 결과의 정상 저장 방지

## 주요 코드 위치

| 경로 | 역할 |
|---|---|
| `src/news_classifier/cli.py` | 터미널 명령과 실행 옵션 처리 |
| `src/news_classifier/pipeline.py` | 수집부터 최종 판단까지의 전체 흐름과 실행 상태 관리 |
| `src/news_classifier/collectors/` | RSS 수집과 선택적 본문 보강 |
| `src/news_classifier/classifiers/`, `src/news_classifier/rules/` | AI 분류, 규칙 근거 계산과 최종 판단 |
| `src/news_classifier/storage/` | CSV와 SQLite 저장 |
| `src/news_classifier/dashboard_streamlit.py` | CSV 결과 시각화 |
| `tests/`, `scripts/` | 자동 테스트, 실제 뉴스 평가와 신뢰도 기준 보정 도구 |

## 검증과 한계

```powershell
python -m pytest -q
```

- 자동 테스트를 통한 정제, 중복 제거, 규칙 판단, 실패 처리와 저장 동작 검증
- 사람이 확정한 정답을 이용한 모델 단독, 규칙 단독, 두 방식 조합의 비교 평가 기능 구현
- 현재 실제 뉴스 분류 정확도 미검증, 자동 테스트 통과와 실제 정확도는 별개의 검증
- 실제 평가에는 사람이 확정한 정답과 개발용/평가용 데이터 분리 필요, 추천 라벨의 정답 사용 불가
- 기본 분류 입력은 기사 제목과 설명 및 선택적 본문이며, 검색어에 의한 판단 편향 방지를 위해 검색어의 모델 입력 제외
- 제목과 언론사명만 반복하는 설명의 모델 입력 제외, 긴 제목/설명 때문에 본문이 잘리지 않도록 입력 길이 분배. 원본 제목과 설명의 CSV 보존
- 현재 주제 정의에 맞지 않는 뉴스의 기타/무관 또는 검토 처리 필요. 검색어마다 여러 주제의 출현이나 균등 분포를 강제하지 않고 기사 내용에 따른 분류
- 유사 기사 묶기는 제목, 설명, 발행일을 사용하는 보수적인 규칙 기반 추정이며, 실제 동일 사건 여부의 완전한 판별은 미보장. 사이트에 따른 본문 수집 실패 가능

동일 사건 묶기 기준: 최대 48시간 이내 발행, 충분한 제목 길이와 단어 수, 기관명 및 수치의 일치, 상반된 표현 제외. 독립적인 설명이 있는 경우 제목 유사도 0.88 이상과 설명 유사도 0.85 이상 확인. RSS 설명이 제목 반복뿐이거나 설명이 없는 경우 제목 유사도 0.95 이상 적용. 발행일이 없거나 짧은 일반 제목인 경우 언론사 간 자동 묶기 제외.
기준값은 보수적인 초기 설정이며 실제 기사로 추가 검증 필요. 그룹의 모든 기사와 비교하여 연쇄적인 오묶음 방지. 과거 CSV는 묶음 정보가 없으면 각 행을 1건으로 조회, 기존 SQLite는 다음 사용 시 컬럼 추가로 기존 행 보존.

### 규칙 보강 절차

1. 다양한 검색어의 후보 기사 수집과 기존 파일 보존. 검색어별 결과의 순환 선택으로 앞 검색어에만 치우친 표본 방지.
2. 기사 링크와 내용을 사람이 확인한 뒤 `gold_label`에 정답 주제, `reviewed_by`에 검토자, `review_status`에 `confirmed` 기록. 같은 사건의 다른 기사에는 동일한 `event_id` 지정.
3. 규칙 후보 확인에 사용한 기사는 `split=development` 지정. 후보 점검 결과의 제안 주제는 정답으로 간주 불가.
4. 기존 규칙 중복과 다른 주제의 사용 사례 확인 후, 서로 다른 사건 최소 2개의 기존 최종 판단 오류 또는 검토를 근거로 규칙 추가. 해당 표현이 실제 점수에 반영되는 기사에서 오류 확인 필수, 같은 사건의 다른 기사 오류로 대체 불가. 출현 횟수나 구문 길이만으로 강한 근거 확정 금지.
5. 규칙 확정 후 후보 선정에 사용하지 않은 별도 사건과 검색어의 평가 기사 준비. `split=evaluation` 지정, 각 split에 현재 전체 정답 주제와 사람이 확인한 `event_id` 포함 필요. 새 규칙 평가에서 검색어 중복 자동 차단.
6. 동일한 모델 출력과 판정 기준값으로 추가 규칙 전후의 검토 비율, 새 자동 분류의 오답과 기존 정답 훼손 비교. 새 표현마다 별도 사건 최소 2개의 올바른 규칙 판단도 확인. 평가 보고서의 관측 검증이며 규칙 자동 활성화 또는 미래 정확도 보장 없음.

사람이 확인한 후보만 별도 JSON으로 제안, `event_rule_candidates` 형식 사용. 아래는 형식 예시이며 실제 증거 또는 바로 적용할 규칙이 아님. 사건 ID는 확인된 development 기사의 `event_id`와 일치 필요.

```json
{
  "schema_version": 1,
  "profile_type": "event_rule_candidates",
  "rules": [{
    "category": "금융/투자",
    "phrase": "사람이 확인한 구체적인 사건 표현",
    "strength": "STRONG",
    "evidence_event_ids": ["개발사건-1", "개발사건-2"]
  }]
}
```

제안 규칙의 독립 평가 통과 시에만 새 적용 파일 생성, 기존 파일 덮어쓰기 차단. 검증 실패 시 보고서만 생성하고 규칙 적용 파일 생성 중단. 모델과 입력 및 기존 규칙 구성이 바뀌면 재평가 필요.

```powershell
# 정답 확인과 독립 사건 및 검색어 분리 완료 후에만 실행
python scripts/evaluate_news.py --dataset data/evaluation/reviewed_news.csv --rules-file data/evaluation/confirmed_candidates.json --export-rule-profile evaluation_results/confirmed_event_rules.json --output-dir evaluation_results/event_rules
# 검증된 공통 규칙을 모든 검색어에 동일하게 적용
news-classifier collect --keyword "원하는 검색어" --rule-profile evaluation_results/confirmed_event_rules.json
```

환경변수 `NEWS_EVENT_RULE_PROFILE`로 적용 파일 지정도 가능. 후보 감사 JSON이나 미검증 제안 JSON의 수집 명령 직접 적용 불가. 적용 파일은 신뢰하는 로컬 설정이며 서명된 인증 자료가 아님. 현재 사람 정답 확인 전이므로 새 사건 규칙의 기본 활성화 없음.

```powershell
python scripts/collect_evaluation_news.py --output data/evaluation/new_candidates.csv
python scripts/review_rule_candidates.py --dataset data/evaluation/new_candidates.csv --output data/evaluation/new_candidate_audit.json
# 사람이 정답 확인을 완료한 데이터만 평가 가능
python scripts/evaluate_news.py --dataset data/evaluation/reviewed_news.csv --split development --output-dir evaluation_results/development
python scripts/evaluate_news.py --dataset data/evaluation/reviewed_news.csv --split evaluation --output-dir evaluation_results/evaluation
```

`기타/무관`은 직접 키워드 규칙 없이 모델 판단 사용. 일반 뉴스 주제는 확인된 development 오류의 최소 2개 독립 사건 근거 확보 후 규칙 활성화 가능, 현재 추가된 확정 규칙 없음. 스포츠와 문화 등의 기사에 대한 기존 무관 문맥 강제 처리 해제. 후보 CSV에는 정답이나 추천 라벨의 자동 입력 없음.
후보 수집은 경제, 산업과 일반 뉴스 검색어를 같은 방식으로 순환 선택. 스포츠와 문화 뉴스의 별도 부정 표본 취급 제거와 `--negative-total` 옵션 폐지.

사건 표현 후보는 기타/무관을 제외한 16개 주제의 국문과 영문으로 구성. 예: 기업공개와 IPO, 신제품 출시와 product launch, 직원 해고와 employee layoffs. 후보 감사에서 기사 속 표현 출현, 기존 규칙 중복, 다른 주제와의 충돌, 취소나 부정 문맥 확인 필요 신호 표시. 회사명과 분야명만으로 주제 확정 없음, 후보의 실제 규칙 적용 없음.
인재 양성과 인력 양성, 생산 능력 확대와 설비 투자도 확인용 후보에 포함. 교육 사업과 기술 연구, 설비 확대와 금융 투자가 함께 등장할 수 있으므로 기사 주요 사건 및 배경 설명의 구분 필요. 표현 출현만으로 해당 주제 확정 또는 규칙 활성화 없음.
확인용 JSON의 `articles_for_human_confirmation`에 원문 링크와 비어 있는 정답, 사건 ID, 검토자 항목 제공. `stored_prediction_not_gold`는 기존 예측으로 정답 아님. 사람이 확인한 값을 평가 CSV에 반영한 뒤 평가 진행, JSON을 그대로 평가 입력으로 사용 불가.

이번 확인 자료: [최신 실행 10건](data/evaluation/event_rule_review_latest_20261004.json), [기존 다분야 후보 180건](data/evaluation/event_rule_review_general_20261004.json). 저장된 CSV의 별도 감사 결과이며 새 뉴스 수집이나 정답 확정 자료가 아님, 원본 CSV 보존.

### 검색어에 독립적인 분류 개선

검색 분야 대신 기사 중심 사건에 따른 판단. 자동차의 연구는 기술개발, 신차 출시는 제품/서비스, 인수는 기업동향으로 구분하는 공통 기준. AI에만 적용하는 별도 분기 없음.

- 모델 내부의 17개 주제 설명 구체화, 기존 주제명 유지와 일반 뉴스 7개 주제 추가
- GPU, 전고체 배터리 같은 기술 종류와 기업명, 국가명 등 기존 일반 명사의 `context_only` 지정. 기사 내용은 유지, 해당 단어의 규칙 점수 및 근거 수 가산 제외
- 기존 사건 구문과 단어 경계, 긴 구문 우선, MODEL/RULE/REVIEW 경로 및 판정 기준값 유지. 사람 검증 없는 새 사건 규칙 추가 없음
- 기사 영역 우선 추출과 중복 문단 제거, 본문 문단이 없으면 기사 구조화 데이터의 본문 확인. 검토 기사 본문 보강은 기본 활성화, 전체 기사 보강은 선택 기능. 접근 차단 등으로 확보 실패 시 제목과 유효 설명만 사용

분류 입력 정책과 주제 설명이 바뀌면 모델 점수도 달라질 가능성. 과거 신뢰도 보정 파일의 자동 재사용 차단, 확인된 development 기사로 보정 파일 재생성 필요. 신뢰도 등급 보정과 최종 판정 기준값 변경은 별개의 작업.
모델 점수는 후보 주제 간 상대 점수이며, 주제 확장에 따른 점수 하락과 검토 비율 증가 가능. 주제 범위 확장만으로 검토 비율 감소 보장 불가, 최종 판정 기준 조정에는 사람이 확인한 정답 자료로 별도 검증 필요.

```powershell
# 여러 분야 후보 수집, 정답과 검토자는 사람 확인 후 작성
python scripts/collect_evaluation_news.py --output data/evaluation/event_candidates.csv --enrich-content
# 개발 단계에서 사용하지 않은 검색어만 별도 후보 수집
python scripts/collect_evaluation_news.py --queries-only --query "항공" --query "보험" --target-total 40 --output data/evaluation/unseen_candidates.csv
# 사람이 검토한 development/evaluation 행을 하나의 CSV에 모은 뒤 비교
python scripts/evaluate_news.py --dataset data/evaluation/reviewed_news.csv --split evaluation --compare-legacy --require-unseen-keywords --output-dir evaluation_results/unseen
```

개발용과 평가용 사건 분리 및 각 split의 현재 전체 정답 주제 확보 필요. 기존 10개 주제의 평가 자료만 있는 경우 새 주제의 사람이 확인한 정답 기사 보충 필요. 동일 사건의 대표 기사 1개 평가 권장. `--require-unseen-keywords`는 두 split의 검색어 중복도 차단하며, 새로운 분야에서의 성능을 보장하는 기능은 아님.
`--compare-legacy`는 같은 모델 가중치와 원문으로 초기 10개 짧은 주제명, 이전 원문 입력과 무관 문맥 규칙을 재현하는 비교. 현재 17개 주제 설명과의 비교이며 확장 전 점수의 그대로 재사용 불가. 검색어별 결과, 다른 주제를 기술개발로 오분류한 건수와 비율, 검토 비율, 잘못된 규칙 보정 함께 확인. 기술개발 비중만 감소시키는 목표는 미사용.
`--compare-short-labels`는 현재와 동일한 17개 주제 및 기사 입력으로 짧은 주제명과 사건 설명의 효과만 비교하는 선택 옵션. 평가 보고서에 두 방식의 모델 및 하이브리드 결과 추가, 기본 모델 설명과 판정 기준의 자동 변경 없음.
기존 CSV와 포트폴리오 PPT 파일의 수정 또는 자동 재분류 없음. 새 주제는 다음 수집부터 적용, 과거 결과는 기존 CSV 기준 조회. 기존 신뢰도 보정 파일은 후보 주제와 설명 불일치로 재사용 차단, 현재 전체 주제를 확인한 development 기사로 재생성 필요.
현재 수정은 동작 검증 단계이며, 사람 정답 자료 부족으로 실제 정확도 향상과 편향 감소 수치 미확정.

### 검토필요 감소를 위한 판정 기준 검증

모든 검색어에 동일한 처리 적용, 특정 검색어 예외 또는 카테고리 분포 강제 없음. 본문이 없는 검토 기사만 정보 보강 시도, 실제 모델 입력 변경 시 최대 1회 재판단. 본문 확보 실패 시 기존 검토 유지, 추가 네트워크 요청과 처리 시간 증가 가능.
Google 뉴스의 기사 식별 정보를 이용한 언론사 URL 조회 후 공개 원문에서 본문 추출. 기사 식별 정보가 없는 경우 중계 페이지 재조회 최대 1회, 각 요청의 시간 제한과 리다이렉트 한도 적용. 중계 페이지 자체의 본문 사용 없음, 로컬 및 사설 주소 거부, 로그인이나 접근 차단 우회 없음. 실패 원인은 터미널 요약과 CSV의 재판단 JSON에 기록.
원문 조회 방식은 비공식 요청 형식으로 변경 가능하며, [참고 구현](https://github.com/SSujitX/google-news-url-decoder)의 기사 URL 조회 흐름 참고. 주소 조회에 성공해도 언론사 접근 차단이나 본문 부재로 실패 가능, 실패 시 검토 유지.
실제 저장 기사 2건의 URL 조회 확인 중 1건 본문 확보와 1건 HTTP 403 차단 확인. 본문을 확보한 1건에서 실제 모델 재판단 후 기존 규칙으로 검토필요에서 금융/투자로 변경 확인. 한 건의 동작 확인이며 사람이 확정한 정답 평가나 전체 검토 비율 개선 수치가 아님.

본문 전용 영역의 안쪽 컨테이너 우선 선택과 중복 문단 제거. HTML 문단이 없는 Arc 형식에서는 현재 페이지 URL 및 제목이 일치하는 공개 기사 JSON의 텍스트 요소만 추출. 스크립트 실행과 관련 기사 데이터 사용, 유료 제한 데이터의 추출 없음. 페이지 구조 변경에 따른 실패 가능성 유지.

같은 저장 기사로 본문 보강 전후 동작 비교 시 아래 명령 사용. 현재 모델과 규칙으로 원래 입력을 다시 판정한 뒤 본문 없는 검토 기사만 보강, 이미 자동 판정한 입력은 유지. 원본 CSV와 기존 출력 파일 덮어쓰기 없음. 보고서의 정답 및 사건 ID는 사람 확인 전 공란 유지.

```powershell
python scripts/compare_body_enrichment.py --input news_analysis_results.csv --output evaluation_results/body_replay.json
```

이번 10건 비교에서는 본문 추출 실패 4건 모두 확보 후 검토 4건에서 0건으로 변경, 기존 자동 판정 6건의 라벨 유지. [동작 비교 기록](data/evaluation/body_replay_20261004_1904.json). 인재 양성 기사의 기업동향 판정 등 주요 사건과 배경 근거가 충돌할 수 있는 사례 포함. 검토 해소는 정답 확인이 아니며, 독립 사건과 미사용 검색어의 사람이 확인한 평가 전 실제 정확도 향상 또는 모든 검색어의 검토 해소 보장 불가. 추가 규칙 활성화와 점수 기준 변경 없음.
같은 10건의 제목과 설명만으로 전체 흐름도 별도 재검증, 최초 검토 8건의 본문 모두 확보 후 최종 검토 1건 확인. 기존 본문을 사용한 부분 보강 비교와 조건이 다르며 원래 자동 판정의 전체 유지 보장 없음. 새 RSS 수집이나 정답 정확도 평가는 아님, 원본 CSV 보존.

```powershell
# 기존 CSV의 검토 사유, 입력 부족, 주제와 검색어별 집계 확인(원본 수정 없음)
python scripts/audit_review_reasons.py --input news_analysis_results.csv
# 사람이 정답과 사건을 확인한 두 split의 자료가 준비된 후 실행
python scripts/calibrate_decision_policy.py --dataset data/evaluation/reviewed_news.csv --output evaluation_results/decision_calibration.json
# 독립 평가까지 통과해 생성된 파일만 사용
news-classifier collect --keyword "원하는 검색어" --decision-profile evaluation_results/decision_calibration.json
```

정답 자료에는 17개 주제와 다양한 검색어의 자동 확정 및 검토 기사 모두 포함 필요. 사람이 확인한 `gold_label`, `review_status=confirmed`, `reviewed_by`, 명시적 `event_id` 필수. 수집 단계의 제목 해시는 독립 사건 확인이 아니며, 같은 사건의 기사들은 사람이 동일한 사건 ID로 정리한 후 개발용과 평가용 사건 및 검색어 분리 필요.

보정 대상은 최종 판정의 모델 점수와 상위 점수 차이 기준이며, 기존 모델 우선 유지와 규칙 보정 기준 유지. 기존 점수 기준 0.50과 점수 차이 0.05뿐 아니라 별도 점수 하한 0.40도 함께 점검. 정답이 확인된 개발 기사에서 관측된 양수 점수와 차이를 후보로 사용, 선택한 점수 기준이 0.40 미만이면 하한도 같은 값으로 조정. 점수 0 또는 1순위와 2순위 동점인 기사는 완화 기준에 의한 모델 자동 분류에서 제외, 명확한 근거에 의한 기존 규칙 보정은 유지. 검색어별 임의 기준 또는 점수 부풀리기 미사용.
개발 자료로 후보 선정 후 미사용 검색어와 별도 사건의 평가 통과 시에만 파일 생성. 기본 안전 목표는 전체 자동 확정과 신규 자동 확정 각각 독립 사건 30개 이상, 정확도 추정의 보수적 하한 0.85 이상. 이 수치는 요구 조건이며 달성한 프로젝트 성능 수치가 아님.
새 자동 확정의 주제별 및 검색어별 관측 정확도도 목표 미달 시 프로필 거부, 전체 평균으로 특정 분야 오답을 가리는 상황 방지. 주제별 표본이 적으면 통계적 불확실성 유지, 모든 주제의 정확도 보장으로 해석 불가.

기본 보정 입력은 제목과 유효 설명(`without_body`), CSV 본문이 있어도 평가 입력에서 제외하며 원본 보존. `--input-mode with_body`는 양쪽 split 모든 기사의 본문 필요. 보정 파일은 검증한 본문 유무와 일치하는 입력에만 적용, 다른 입력에는 기본 기준 사용. 주제와 검색어별 결과도 확인하되 모든 미래 검색어의 정확도 보장 불가.

정답 자료 부족 또는 안전 목표 미달이면 완화된 파일 생성 없이 기본 기준 유지. 환경변수 지정 시 `NEWS_DECISION_CALIBRATION`, 기존 `NEWS_CONFIDENCE_CALIBRATION`은 신뢰도 표시 전용. 과거 CSV와 PPT의 수정 또는 자동 재분류 없음.
보정 명령 완료 시 실제 적용할 세 기준, 수정 전후 검토 비율과 자동 분류 오답 수, 새 자동 분류의 관측 정확도 출력. 새 프로필은 점수 하한까지 저장하는 버전 2 형식, 기존 버전 1 프로필은 원래 하한을 유지하며 검증 조건 일치 시 사용 가능.
현재 사람이 확인한 정답 자료가 없으므로 낮은 기준의 기본 적용과 실제 검토 비율 감소 확인은 미완료. 위의 일반 수집 명령만 실행하면 검증되지 않은 기준으로 자동 완화되지 않음.
