# 뉴스 수집 및 분류기

Python 기반 뉴스 수집 및 분류기 프로젝트입니다.

Google News RSS에서 키워드 기반 뉴스를 수집하고, Hugging Face zero-shot 분류 모델과 규칙 기반 보정 로직을 활용하여 뉴스 카테고리를 분류합니다.  
분류 결과는 CSV 파일로 저장하여 Excel로 확인할 수 있고, SQLite DB에 누적 저장하여 필요할 때 다시 조회할 수 있습니다.

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
- 최종 카테고리, 신뢰도, 보정 여부 저장
- CSV 결과 저장
- SQLite 누적 저장
- 카테고리별 기사 수 요약 출력
- 낮은 신뢰도 기사 재검토 가능

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
├─ pyproject.toml
├─ .env.example
├─ ARCHITECTURE.md
├─ LINE_COUNTS.json
│
├─ src/
│  └─ news_classifier/
│     ├─ cli.py
│     ├─ pipeline.py
│     ├─ service.py
│     ├─ config.py
│     ├─ models.py
│     ├─ scheduler.py
│     ├─ dashboard_streamlit.py
│     │
│     ├─ collectors/
│     ├─ classifiers/
│     ├─ dedup/
│     ├─ features/
│     ├─ storage/
│     ├─ exporters/
│     ├─ reporting/
│     ├─ rules/
│     └─ utils/
│
├─ tests/
├─ data/
└─ scripts/
```

---

## 주요 파일 설명

| 파일 | 설명 |
|---|---|
| `run_news.ps1` | 실행 환경 설정 및 뉴스 수집·분류 CLI 실행 |
| `src/news_classifier/cli.py` | 터미널 명령어 해석 및 뉴스 수집·분류 실행 진입점 |
| `src/news_classifier/service.py` | 수집기, 분류기, 규칙 엔진, 중복 제거기 조립 및 파이프라인 구성 |
| `src/news_classifier/pipeline.py` | 뉴스 수집, 중복 제거, 분류, 후처리까지의 전체 처리 흐름 실행 |
| `src/news_classifier/collectors/google_rss.py` | Google News RSS 기반 키워드 뉴스 수집 및 기사 정보 추출 |
| `src/news_classifier/classifiers/zero_shot_classifier.py` | Hugging Face zero-shot 모델 기반 뉴스 1차 카테고리 분류 |
| `src/news_classifier/classifiers/rule_engine.py` | 카테고리별 키워드 매칭 점수 계산 및 규칙 기반 보정 근거 생성 |
| `src/news_classifier/classifiers/postprocessor.py` | 모델 점수, margin, 규칙 점수 기반 최종 카테고리 결정 |
| `src/news_classifier/rules/default_rules.py` | 기본 분류 카테고리 및 키워드 규칙 사전 관리 |
| `src/news_classifier/storage/csv_store.py` | 분류 결과 CSV 저장 및 Excel 확인용 파일 생성 |
| `src/news_classifier/storage/sqlite_store.py` | 분류 결과 SQLite DB 누적 저장 및 재조회 지원 |
| `src/news_classifier/reporting/summary_report.py` | 전체 기사 수, 낮은 신뢰도 기사 수, 규칙 보정 수, 카테고리별 개수 요약 출력 |

---

## 실행 환경

- Python 3.12 이상 권장
- Windows PowerShell 기준
- VS Code 터미널 기준
- 기본 실행은 CPU에서 가능
- NVIDIA GPU와 CUDA 지원 PyTorch가 있으면 zero-shot 분류 모델은 GPU 사용

---

## 실행 방법

### 1. 프로젝트 폴더로 이동

```powershell
cd C:\news_classifier_expanded_project\news_classifier_expanded
```

---

### 2. 패키지 설치

처음 실행하거나 패키지가 없을 때만 설치

![가상환경 실행 예시](image-2.png)

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

이미 설치된 경우 생략 가능

---

### 3-1. CPU 환경일 때 실행

GPU 설정 없이 기본 환경으로 실행하는 방법

패키지 설치

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

뉴스 수집 및 분류 실행

```powershell
.\run_news.ps1
```

실행 로그 예시

```text
loading classifier model=... device=cpu
```

CPU 환경은 별도 GPU 설정 없이 실행 가능

---

### 3-2. GPU 환경일 때 실행

NVIDIA GPU와 CUDA 지원 PyTorch가 있는 환경에서 zero-shot 분류 모델을 GPU로 실행하는 방법

프로젝트 폴더 이동

```powershell
cd C:\news_classifier_expanded_project\news_classifier_expanded
```

CUDA PyTorch 설치 여부 확인

```powershell
..\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

정상 출력 예시

`True`와 NVIDIA GPU 이름이 나오면 바로 GPU 실행 가능

뉴스 수집 및 분류 실행(torch 설치되어 있으면 해당 명령만 입력)

```powershell
.\run_news.ps1
```

`device=cuda`가 보이면 GPU 실행 상태
`torch` 버전에 `+cpu`가 보이거나 `torch.cuda.is_available()` 결과가 `False`이면 CPU 전용 PyTorch 상태

CUDA 지원 PyTorch 설치

```powershell
..\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
..\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

설치 후 CUDA 인식 재확인

```powershell
..\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

GPU가 없거나 CUDA 지원 PyTorch가 설치되지 않은 경우 자동 CPU 실행

---

### 5. 공통 실행 세부 내용

```powershell
.\run_news.ps1
```

스크립트 내부 처리 내용

```powershell
$env:PYTHONPATH = "src"
..\.venv\Scripts\python.exe -m news_classifier.cli collect --limit 10 --csv out.csv --sqlite news.db
```

실행 후 터미널에 검색 키워드 입력 문구 표시

```text
검색 키워드를 입력하세요:
```

예시 입력

```text
AI 반도체
```

---

### 6. 결과 파일 확인

실행 완료 후 프로젝트 폴더에 결과 파일 생성

| 파일 | 용도 |
|---|---|
| `out.csv` | Excel 기반 분류 결과 확인 |
| `news.db` | SQLite 기반 누적 저장 및 재조회 |

CSV 결과 열기

```powershell
ii .\out.csv
```

![CSV 결과 예시](image-1.png)

---

### 7. 대시보드로 결과 확인

Streamlit 대시보드 실행

```powershell
..\.venv\Scripts\Activate.ps1
$env:PYTHONPATH="src"
streamlit run src\news_classifier\dashboard_streamlit.py
```

대시보드의 CSV 경로 입력칸에 `out.csv` 입력

---

## 다른 키워드로 실행하기

실행 스크립트를 다시 실행한 뒤 입력 문구에 다른 키워드 입력

```powershell
.\run_news.ps1
```

```text
검색 키워드를 입력하세요: 생성형 AI
```

다른 예시 입력

```text
검색 키워드를 입력하세요: 반도체 공급망
```

스크립트 실행 결과는 `out.csv`에 새로 저장되고, `news.db`에는 같은 DB 파일로 누적 저장

---

## SQLite 저장 결과 확인

최근 저장된 뉴스 20개를 터미널에서 확인합니다.

```powershell
python -c "import sqlite3, pandas as pd; conn=sqlite3.connect('news.db'); df=pd.read_sql_query('SELECT title, source, final_category, confidence_level, rule_applied, published_at FROM classified_news ORDER BY id DESC LIMIT 20', conn); print(df.to_string(index=False))"
```

카테고리별 기사 수를 확인합니다.

```powershell
python -c "import sqlite3, pandas as pd; conn=sqlite3.connect('news.db'); df=pd.read_sql_query('SELECT final_category, COUNT(*) AS count FROM classified_news GROUP BY final_category ORDER BY count DESC', conn); print(df.to_string(index=False))"
```

낮은 신뢰도 기사만 확인합니다.

```powershell
python -c "import sqlite3, pandas as pd; conn=sqlite3.connect('news.db'); df=pd.read_sql_query(\"SELECT title, source, final_category, confidence_level, rule_reason FROM classified_news WHERE confidence_level='낮음' ORDER BY id DESC LIMIT 20\", conn); print(df.to_string(index=False))"
```

---

## 실행 예시

```powershell
cd C:\news_classifier_expanded_project\news_classifier_expanded
.\run_news.ps1
```

키워드 입력 예시

```text
검색 키워드를 입력하세요: AI 반도체
```

실행 결과 예시는 다음과 같습니다.

```text
CSV saved: out.csv

# 뉴스 분류 요약
- 전체 기사 수: 10
- 낮은 신뢰도 기사 수: 2
- 규칙 보정 적용 기사 수: 1

## 카테고리별 기사 수
- 기술개발: 8
- 기업동향: 1
- 시장/투자: 1
```

## 전체 처리 과정

1. 사용자가 `run_news.ps1` 실행
2. 스크립트가 `PYTHONPATH` 설정 후 CLI 실행
3. 사용자가 터미널 입력 문구에 검색 키워드 입력
4. cli.py가 입력값과 실행 옵션 해석
5. service.py가 수집기, 분류기, 규칙 엔진, 저장소 객체 조립
6. pipeline.py가 전체 처리 흐름 실행
7. google_rss.py가 Google News RSS에서 뉴스 수집
8. 중복 뉴스 제거 후 필요하면 article_scraper.py로 본문 보강
9. zero_shot_classifier.py가 Hugging Face 모델로 1차 분류 수행
10. rule_engine.py와 postprocessor.py가 모델 점수, margin, 키워드 규칙을 바탕으로 최종 카테고리 결정
11. csv_store.py와 sqlite_store.py가 결과를 CSV와 SQLite에 저장

## GitHub 업로드

```powershell
git init
git add .
git status
git commit -m "뉴스 수집 및 분류기 프로젝트 초기 커밋"
git branch -M main
git remote add origin https://github.com/Yang0Jin0Woo/news-collection-classification_python.git
git push -u origin main
```

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
- 최종 카테고리, 신뢰도, 규칙 보정 여부 저장
- 낮은 신뢰도 기사 재검토 기능 추가
- 카테고리별 기사 수 요약 조회 기능 추가
- 전처리, 중복 제거, 규칙 보정, 파이프라인 검증 테스트 추가
- 카테고리별 샘플 뉴스 케이스 추가
