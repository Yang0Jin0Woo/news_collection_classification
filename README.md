# 뉴소트(NewSort)

검색어에 맞는 뉴스를 수집하고, 기사 내용을 17개 주제로 분류하는 Python 프로젝트

AI 예측과 기사 속 규칙 근거를 비교한 최종 판단, 결과와 판단 사유의 CSV 저장 및 대시보드 조회

사용 도구: Python, PyTorch, Transformers, Pandas, BeautifulSoup, Streamlit, SQLite, Pytest

## 핵심 기능

- 뉴스 수집: Google News RSS의 최근 7일 기사 수집과 제목 정리
- 중복 묶기: 동일 사건으로 추정되는 유사 기사 묶기, 대표 기사 표시와 나머지 원문 링크 보존
- 주제 분류: 추가 학습 없이 다국어 모델 mDeBERTa 사용, 기사 제목과 설명 및 확보한 본문으로 판단
- 규칙 판단: 모델 점수와 상위 두 주제의 점수 차이, 기사 속 표현을 비교한 모델 유지 또는 규칙 보정
- 검토 기사 재판단: 본문이 없는 검토 기사의 원문 확보 시도와 최대 1회 재분류, 불확실한 결과의 검토 유지
- 결과 확인: CSV 저장, Streamlit 대시보드 조회, 선택적 SQLite 누적 저장

규칙의 단어 경계 확인과 긴 구문 우선 탐색, 같은 표현의 중복 가산 방지

강한 근거 2점과 일반 근거 1점 적용, 기업명과 분야명만으로 규칙 점수 가산 제외

분류 주제: 기술개발, 제품/서비스, 기업동향, 생산/공급망, 정책/규제, 금융/투자, 시장/산업, 노동/노사, 국제/통상, 교육/취업, 사회, 정치, 문화/연예, 스포츠, 건강/의료, 생활/환경, 기타/무관

## 처리 흐름

검색어 입력 → RSS 수집 → 정제와 중복 묶기 → AI 분류 → 규칙 판단 → 필요한 기사 본문 보강과 재판단 → 저장과 결과 확인

![뉴스 수집부터 분류와 저장까지의 아키텍처](docs/architecture.png)

## 실행 방법

Python 3.11~3.13과 인터넷 연결 필요, Windows PowerShell 기준

실행 위치: `pyproject.toml`이 있는 `news_classifier_expanded` 폴더

가상환경은 패키지를 따로 설치하는 공간, 상위 폴더의 `.venv` 하나만 사용

- 처음 사용: 1번 설치 → 3번 실행
- 새 터미널에서 재사용: 2번 활성화 → 3번 실행

### 1. 최초 설치

상위 `.venv`가 없을 때 한 번만 실행, 설치 완료 후 3번 진행

```powershell
python -m venv ..\.venv
..\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

### 2. 환경 활성화

설치된 환경을 새 터미널에서 활성화, 최초 설치 직후나 이미 활성화된 경우 생략

```powershell
..\.venv\Scripts\Activate.ps1
```

### 3. 실행

`AI 반도체`를 원하는 검색어로 변경 후 수집 실행

```powershell
news-classifier collect --keyword "AI 반도체"
```

수집 완료 후 대시보드 실행

```powershell
news-classifier dashboard
```

Local URL로 접속, 종료는 `Ctrl+C` 입력

기본 최대 10건 수집, 중복 묶기 후 대표 기사 수 감소 가능

최초 모델 다운로드와 로딩 시간 소요, CUDA 사용 가능 시 GPU 선택 및 그 외 CPU 사용

macOS와 Linux의 활성화 명령은 `source ../.venv/bin/activate` 사용

기본 실행에 `.env` 파일 불필요, 설정 변경 시 `.env.example` 참고

### 주요 옵션

| 옵션 | 용도 |
|---|---|
| `--limit 20` | 최대 수집량 지정, 허용 범위 1~100 |
| `--csv result.csv` | CSV 저장 경로 지정, 기존 결과 보관 시 별도 파일명 사용 |
| `--sqlite` | SQLite에도 저장, 기본 파일 `news_analysis.db` |
| `--enrich-content` | 최초 분류 전 전체 기사 본문 확보 시도 |
| `--no-review-enrichment` | 검토 기사 본문 보강과 재판단 비활성화 |

전체 옵션 확인 명령: `news-classifier collect --help`

## 실행 예시

검색어 입력과 수집 명령 화면

![뉴스 수집과 분류 명령 입력 화면](image.png)

분류 결과와 실행 상태 화면, 이전 주제 체계의 예시이며 현재 결과와 차이 가능

![뉴스 분류 결과와 실행 상태 출력 화면](image-1.png)

## 결과 확인

- CSV: 기본 파일 `news_analysis_results.csv`, Excel에서 확인 가능, 실행마다 해당 파일 덮어쓰기
- SQLite: 옵션 지정 시 결과 누적 저장, 같은 검색어와 제목 및 언론사의 기사 갱신
- 대시보드: 최종 주제(`final_category`) 기준 대표 기사 수 집계, 실제 결과에 있는 주제만 표시

최종 판단의 출처를 나타내는 `decision_source`의 정상 판정 경로 3종

| 값 | 의미 |
|---|---|
| `MODEL` | 모델 예측 유지 |
| `RULE` | 규칙 근거로 최종 주제 결정 |
| `REVIEW` | 판단 근거 부족이나 주제 간 충돌로 사람 검토 필요 |

`rule_reason`에 판단 사유, `review_reclassification`에 본문 확보와 재판단 과정 기록

`model_confidence`는 모델 예측의 참고 점수이며 최종 결과의 정답 확률 보장 없음

정상 실행과 검색 결과 0건은 종료 코드 `0`, 수집과 모델 오류는 결과 저장 중단 및 종료 코드 `1`

검색 결과 0건은 열 이름만 있는 CSV 저장, 오류 발생 시 터미널 메시지 확인

## 주요 코드

| 경로 | 역할 |
|---|---|
| `src/news_classifier/cli.py` | 수집과 대시보드 명령 처리 |
| `src/news_classifier/pipeline.py` | 전체 처리 흐름과 실행 상태 관리 |
| `src/news_classifier/collectors/` | RSS 수집과 원문 본문 추출 |
| `src/news_classifier/classifiers/`, `src/news_classifier/rules/` | AI 분류와 규칙 판단 |
| `src/news_classifier/storage/`, `src/news_classifier/dashboard_streamlit.py` | 결과 저장과 대시보드 |
| `tests/`, `scripts/` | 자동 테스트와 기사 평가 도구 |

## 검증과 한계

```powershell
python -m pytest -q
```

- 자동 테스트로 정제, 중복 묶기, 규칙 판단, 오류 처리와 저장 동작 확인
- 실제 뉴스 분류 정확도는 사람 정답 자료 부족으로 미검증, 검토 감소와 정확도 향상은 별개
- 검색어는 뉴스 수집에만 사용하고 모델 입력에서 제외, 검색어별 예외나 주제 비율 강제 없음
- 제목 유사도와 발행일 등을 이용한 중복 묶기로 동일 사건의 완전한 판별 미보장
- 언론사 접근 차단과 페이지 구조에 따른 본문 확보 실패 가능, 본문 보강에 추가 시간 소요
- 기사 배경 설명을 주된 사건의 근거로 오인할 가능성, 불확실한 결과의 사람 확인 필요
- 새 규칙과 점수 기준 변경은 사람이 확인한 정답, 별도 사건과 검색어의 평가 후 적용

검토 원인 분석은 `scripts/audit_review_reasons.py`, 본문 보강 전후 비교는 `scripts/compare_body_enrichment.py` 사용

[본문 보강 동작 비교 기록](data/evaluation/body_replay_20261004_1904.json)은 같은 저장 기사의 재실행 결과이며 실제 정확도 평가 자료와 구분

### 정답 확인과 재검증

검토 기사와 자동 분류 기사 모두 원문 확인 → 주요 사건과 배경 표현 기록 → 확인된 오분류 원인 수정 → 별도 사건과 검색어로 재검증

`scripts/review_rule_candidates.py`의 확인 양식에 정답과 주요 사건 기록, 규칙 단어의 위치와 주변 문장은 확인용 자료로만 사용

`scripts/evaluate_news.py`에서 검토 비율, 자동 분류 오답, 기존 정답 훼손 비교

사용자가 확인한 [개발 기사 2건](data/evaluation/user_confirmed_context_20261004.json)으로 배경 근거의 자동 보정 보류안 비교, 독립 평가 전 기본 실행에 적용 없음

```powershell
python scripts/compare_rule_context.py --dataset data/evaluation/user_confirmed_context_20261004.json --output evaluation_results/rule_context_comparison.json
```

수정안은 제목, 설명과 본문 첫 두 문장의 근거를 추가 확인하는 실험용 처리, 주요 사건의 완전한 판별 미보장
사람이 확인한 별도 사건과 검색어, 기존 자동 정답 기사까지 추가 평가 후 적용 여부 결정
