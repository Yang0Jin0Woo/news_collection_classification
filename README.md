# 뉴소트(NewSort)

Python 기반 뉴스 수집 및 분류기 프로젝트입니다.

Google News RSS에서 키워드 기반 뉴스를 수집하고, Hugging Face zero-shot 분류 모델과 규칙 기반 보정 로직을 활용하여 뉴스 카테고리를 분류합니다.  
분류 결과는 CSV 파일로 저장하여 Excel로 확인할 수 있고, SQLite DB에 누적 저장하여 필요할 때 다시 조회할 수 있습니다.

## 아키텍처

![아키텍처 구조](docs/architecture.png)

### 핵심 흐름

**[키워드 입력] → [뉴스 수집] → [정제·중복 제거] → [Zero-shot 분류 (mDeBERTa)] → [규칙 보정] → [출력/대시보드]**

1. **유연한 분류 방식:** 사전학습된 제로샷 모델을 활용해 별도 재학습 없이 후보 카테고리 분류
2. **검증 가능한 구조:** 수동 라벨이 확정된 실제 뉴스로 모델·규칙·하이브리드 방식 비교 지원
3. **설명 가능한 보정:** 모델 점수와 규칙 적용 사유를 함께 저장해 최종 판단 근거 확인

---

![실행 결과 예시](image.png)

![CSV 결과 예시](image-1.png)

---

## 주요 기능

- Google News RSS 기반 뉴스 수집
- 뉴스 제목 및 설명 텍스트 전처리
- 중복 뉴스 제거
- Hugging Face zero-shot 분류 모델 기반 1차 분류
- 모델 점수와 top1-top2 margin 기반 신뢰도 판단
- 규칙 기반 키워드 보정
- 모델 신뢰도와 최종 판단 출처를 분리해 저장
- 최종 카테고리, 판단 상태, 사람 검토 필요 여부 저장
- CSV 결과 저장
- SQLite 누적 저장
- 카테고리별 기사 수 요약 출력
- 낮은 모델 신뢰도 및 검토필요 기사 재검토 가능

결과 데이터에서 `model_confidence`는 `model_category`에만 해당합니다.
최종 카테고리의 판단 근거는 `decision_source`로 구분합니다.

- `MODEL`: 모델 예측을 최종 결과로 채택
- `RULE`: 규칙이 최종 결과를 결정
- `REVIEW`: 근거 부족으로 사람 검토 필요
- `ERROR`: 모델 실행 또는 분류 실패

`final_decision_status`는 `DECIDED`, `REVIEW_REQUIRED`, `ERROR` 중 하나이며,
`review_required`는 사람이 확인해야 하는 결과인지 나타냅니다.

파이프라인은 단순 목록 대신 `PipelineResult`를 반환합니다.

- `status`: `SUCCESS`, `NO_RESULTS`, `NETWORK_ERROR`, `COLLECTION_ERROR`, `MODEL_ERROR`
- `results`: 정상적으로 완료된 개별 분류 결과(전체 실패 시 부분 결과 포함)
- `errors`: 실패 단계, 오류 코드, 오류 메시지
- `statistics`: 요청·수집·중복 제거·분류·규칙 보정·검토 필요 건수

`NO_RESULTS`는 검색 결과가 없는 정상 상태로 처리하며 헤더만 있는 CSV를 저장합니다.
네트워크·RSS 파싱·모델 분류 실패와 모델 출력 개수 불일치는 결과를 저장하지 않고
CLI 종료 코드 `1`을 반환합니다.

---

## 사용 기술

- Python
- Google News RSS
- Hugging Face Transformers
- Zero-shot Classification
- Pandas
- BeautifulSoup
- SQLite
- CSV
- Pytest

---

## 프로젝트 구조

```text
news_classifier_expanded/
├─ README.md
├─ requirements.txt
├─ requirements.lock
├─ pyproject.toml
├─ .env.example
├─ run_news.ps1
│
├─ src/
│  └─ news_classifier/
│     ├─ cli.py
│     ├─ evaluation.py
│     ├─ pipeline.py
│     ├─ service.py
│     ├─ config.py
│     ├─ models.py
│     ├─ dashboard_streamlit.py
│     │
│     ├─ collectors/
│     ├─ classifiers/
│     ├─ dedup/
│     ├─ storage/
│     ├─ reporting/
│     ├─ rules/
│     └─ utils/
│
├─ tests/
├─ data/
│  └─ evaluation/
└─ scripts/
```

---

## 주요 파일 설명

| 파일 | 설명 |
|---|---|
| `run_news.ps1` | 설치된 `news-classifier` 명령을 호출하는 Windows 보조 스크립트 |
| `pyproject.toml` | 패키지 의존성, Python 범위, CLI 진입점 정의 |
| `requirements.lock` | 하위 패키지까지 고정한 재현용 의존성 목록 |
| `src/news_classifier/cli.py` | 터미널 명령어 해석 및 뉴스 수집·분류 실행 진입점 |
| `src/news_classifier/service.py` | 수집기, 분류기, 규칙 엔진, 중복 제거기 조립 및 파이프라인 구성 |
| `src/news_classifier/pipeline.py` | 전체 처리 흐름과 성공·빈 결과·수집·모델 실패 상태 반환 |
| `src/news_classifier/collectors/google_rss.py` | Google News RSS 수집 및 네트워크 실패·빈 결과·파싱 오류 구분 |
| `src/news_classifier/classifiers/zero_shot_classifier.py` | Hugging Face zero-shot 모델 기반 뉴스 1차 카테고리 분류 |
| `src/news_classifier/classifiers/rule_engine.py` | 카테고리별 키워드 매칭 점수 계산 및 규칙 기반 보정 근거 생성 |
| `src/news_classifier/classifiers/postprocessor.py` | 모델 점수, margin, 규칙 점수 기반 최종 카테고리 결정 |
| `src/news_classifier/rules/default_rules.py` | 기본 분류 카테고리 및 키워드 규칙 사전 관리 |
| `src/news_classifier/storage/csv_store.py` | 분류 결과 CSV 저장 및 빈 결과의 고정 헤더 생성 |
| `src/news_classifier/storage/sqlite_store.py` | 분류 결과 SQLite DB 누적 저장 및 재조회 지원 |
| `src/news_classifier/reporting/summary_report.py` | 전체 기사 수, 낮은 모델 신뢰도, 규칙 보정·검토필요·오류 수 요약 출력 |
| `src/news_classifier/evaluation.py` | 정확도·거시 평균 F1·카테고리별 정밀도와 재현율·혼동행렬 계산 |
| `scripts/collect_evaluation_news.py` | 수동 라벨링에 사용할 실제 뉴스 후보 수집 |
| `scripts/evaluate_news.py` | 모델 단독·규칙 단독·하이브리드 분류 성능 비교 |

---

## 실제 뉴스 분류 성능 평가

테스트 통과 개수와 실제 분류 정확도는 다릅니다. 이 프로젝트는 사람이
확정한 실제 뉴스 라벨만 성능 평가에 사용하며, 검색어 기반 추천 라벨은
정답으로 취급하지 않습니다.

### 1. 실제 뉴스 후보 수집

```bash
python scripts/collect_evaluation_news.py --output data/evaluation/real_news_candidates.csv --target-per-label 20
```

현재 후보 파일에는 9개 카테고리별 20건씩 총 180건의 실제 뉴스가
들어 있습니다.

### 2. 수동 라벨 확정

`data/evaluation/real_news_candidates.csv`의 다음 열을 사람이 직접 작성합니다.

- `gold_label`: 사람이 확정한 최종 카테고리
- `review_status`: 검토 완료 시 `confirmed`
- `reviewed_by`: 검토자 이름
- `split`: 규칙 개발용은 `development`, 최종 평가는 `evaluation`

같은 사건을 다룬 유사 기사는 동일한 분할에 넣어 정보 누출을 방지합니다.
확정되지 않은 데이터가 포함되면 평가 명령은 실행을 중단합니다.

### 3. 세 가지 분류 방식 비교

```bash
python scripts/evaluate_news.py --dataset data/evaluation/real_news_candidates.csv --split evaluation --output-dir evaluation_results
```

평가 결과에는 다음 내용이 생성됩니다.

- 모델 단독·규칙 단독·하이브리드 방식의 정확도와 거시 평균 F1
- 카테고리별 정밀도·재현율·F1
- 방식별 혼동행렬
- 평가 데이터, 모델 이름, 카테고리 목록

수동 라벨 검토가 끝나기 전에는 정확도 수치를 README나 포트폴리오에
표시하지 않습니다.

---

## 실행 환경

- Python 3.11~3.13
- 최초 실행 시 모델 다운로드를 위한 인터넷 연결 필요

---

## 실행 방법

상위 폴더에서 터미널을 열었다면 프로젝트 폴더로 이동합니다.

```powershell
cd .\news_classifier_expanded
```

터미널 경로가 이미 `news_classifier_expanded`로 끝난다면 이 명령은 생략합니다.
이후 저장소 최상위 폴더에서 가상환경을 생성합니다.

```bash
python -m venv .venv
```

가상환경을 활성화합니다.

```powershell
# Windows
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

패키지를 설치하고 뉴스를 수집·분류합니다.

```bash
python -m pip install -e .
news-classifier collect --keyword "AI 반도체"
```

정상 실행되면 프로젝트 폴더에 `news_analysis_results.csv`가 생성됩니다.
테스트는 다음 명령으로 확인합니다.

```bash
pytest -q
```

## 전체 처리 과정

1. 사용자가 `news-classifier collect --keyword "검색어"` 실행
2. cli.py가 `.env`, 입력값과 실행 옵션 해석
3. service.py가 수집기, 분류기, 규칙 엔진, 저장소 객체 조립
4. pipeline.py가 전체 처리 흐름 실행
5. google_rss.py가 Google News RSS에서 뉴스 수집
6. 중복 뉴스 제거 후 필요하면 article_scraper.py로 본문 보강
7. zero_shot_classifier.py가 고정 revision 모델로 1차 분류 수행
8. rule_engine.py와 postprocessor.py가 모델 점수, margin, 키워드 규칙을 바탕으로 최종 카테고리 결정
9. csv_store.py가 CSV를 저장하고, 옵션 지정 시 sqlite_store.py가 SQLite에도 저장

## 프로젝트 개선 내용

초기 버전은 단일 Python 스크립트에서 뉴스 수집, 분류, 규칙 보정, CSV 저장을 처리하는 구조였습니다.

이후 수집기, 분류기, 규칙 보정, 저장소, 리포트, 테스트 코드, 샘플 뉴스 데이터로 기능을 분리했습니다. 또한 SQLite 저장 기능을 추가하여 뉴스 분류 결과의 누적 저장 및 재조회가 가능하도록 개선했습니다.

주요 개선 사항은 다음과 같습니다.

- 단일 스크립트에서 모듈형 프로젝트 구조로 개선
- Google News RSS 수집 로직 분리
- zero-shot 분류 모델 처리 로직 분리
- 모델 점수와 margin 기반 신뢰도 판단 추가
- 카테고리별 키워드 규칙 사전 확장
- 규칙 기반 키워드 보정 로직 강화
- CSV 저장과 SQLite 누적 저장 기능 분리
- 최종 카테고리와 모델 신뢰도를 구분해 저장
- 판단 출처와 사람 검토 필요 여부 저장
- 카테고리별 기사 수 요약 조회 기능 추가
- 전처리, 중복 제거, 규칙 보정, 파이프라인 검증 테스트 추가
- 9개 카테고리의 기대 라벨을 직접 비교하는 회귀 테스트 추가
