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

## 폴더 및 파일 설명

| 경로 | 설명 |
|---|---|
| `src/news_classifier/cli.py` | 터미널 명령어를 받아 뉴스 수집 및 분류를 실행하는 진입점 |
| `src/news_classifier/pipeline.py` | 뉴스 수집, 중복 제거, 분류, 보정, 저장 흐름을 연결하는 핵심 파이프라인 |
| `src/news_classifier/service.py` | 파이프라인 기능을 서비스 단위로 묶어 실행하기 위한 파일 |
| `src/news_classifier/models.py` | 뉴스 데이터와 분류 결과에 사용되는 데이터 구조 정의 |
| `src/news_classifier/config.py` | 모델명, 라벨, 실행 옵션 등 설정값 관리 |
| `src/news_classifier/collectors/` | Google News RSS 수집 및 기사 데이터 추출 기능 |
| `src/news_classifier/classifiers/` | zero-shot 분류, 규칙 보정, 신뢰도 판단 기능 |
| `src/news_classifier/dedup/` | 중복 뉴스 제거 기능 |
| `src/news_classifier/features/` | 뉴스 텍스트의 특징 추출 기능 |
| `src/news_classifier/storage/` | CSV 및 SQLite 저장 기능 |
| `src/news_classifier/exporters/` | Excel 등 결과 내보내기 기능 |
| `src/news_classifier/reporting/` | 카테고리별 기사 수, 낮은 신뢰도 기사 수 등 요약 리포트 생성 |
| `src/news_classifier/rules/` | 카테고리별 키워드 규칙 사전 |
| `src/news_classifier/utils/` | 텍스트 정제, HTTP 요청, 날짜 처리, 검증 등 공통 유틸 |
| `tests/` | 전처리, 중복 제거, 규칙 보정, 파이프라인 동작 검증 테스트 코드 |
| `data/` | 분류 검증용 샘플 뉴스 케이스 |
| `scripts/` | 실행 보조 스크립트 |
| `requirements.txt` | 프로젝트 실행에 필요한 Python 패키지 목록 |
| `ARCHITECTURE.md` | 프로젝트 구조와 설계 설명 문서 |
| `LINE_COUNTS.json` | 파일별 행 수 정보 |

---

## 실행 환경

- Python 3.12 이상 권장
- Windows PowerShell 기준
- VS Code 터미널 기준

---

## 실행 방법

### 1. 프로젝트 폴더로 이동

```powershell
cd C:\Users\juyan\Downloads\news_classifier_expanded_project\news_classifier_expanded
```

### 2. 가상환경 생성 및 활성화

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

정상적으로 활성화되면 터미널 앞에 `(.venv)`가 표시됩니다.

### 3. 패키지 설치

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Python 경로 설정

```powershell
$env:PYTHONPATH="src"
```

### 5. CSV 파일로 결과 저장

```powershell
python -m news_classifier.cli collect --keyword "AI 반도체" --limit 10 --csv out.csv
```

실행 후 `out.csv` 파일이 생성됩니다.  
CSV 파일은 Excel로 열어 결과를 확인할 수 있습니다.

```powershell
ii .\out.csv
```

### 6. CSV와 SQLite에 동시에 저장

```powershell
python -m news_classifier.cli collect --keyword "AI 반도체" --limit 10 --csv out.csv --sqlite news.db
```

실행 후 아래 파일이 생성됩니다.

| 파일 | 용도 |
|---|---|
| `out.csv` | Excel로 결과 확인 |
| `news.db` | 뉴스 분류 결과 누적 저장 및 재조회 |

---

## 다른 키워드로 실행하기

키워드를 바꾸면 다른 주제의 뉴스도 수집할 수 있습니다.

```powershell
python -m news_classifier.cli collect --keyword "생성형 AI" --limit 10 --csv generative_ai.csv --sqlite news.db
```

```powershell
python -m news_classifier.cli collect --keyword "반도체 공급망" --limit 10 --csv semiconductor_supply.csv --sqlite news.db
```

SQLite 파일명을 동일하게 `news.db`로 지정하면 여러 키워드의 결과를 하나의 DB에 누적 저장할 수 있습니다.

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

## 테스트 실행

```powershell
python -m pytest
```

---

## 실행 예시

```powershell
cd C:\Users\juyan\Downloads\news_classifier_expanded_project\news_classifier_expanded
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONPATH="src"
python -m news_classifier.cli collect --keyword "AI 반도체" --limit 10 --csv out.csv --sqlite news.db
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

이후 프로젝트를 기능별로 분리하여 수집기, 분류기, 규칙 보정, 저장소, 리포트, 테스트 코드로 구성했습니다. 또한 SQLite 저장 기능을 추가하여 단발성 CSV 확인뿐 아니라 뉴스 분류 결과를 누적 저장하고 필요할 때 다시 조회할 수 있도록 개선했습니다.

주요 개선 사항은 다음과 같습니다.

- 단일 스크립트에서 모듈형 프로젝트 구조로 개선
- Google News RSS 수집 로직 분리
- zero-shot 분류 모델 처리 로직 분리
- 모델 점수와 margin 기반 신뢰도 판단 추가
- 규칙 기반 키워드 보정 로직 강화
- CSV 저장과 SQLite 누적 저장 기능 분리
- 키워드별 뉴스 수집 결과 누적 관리
- 최종 카테고리, 신뢰도, 규칙 보정 여부 저장
- 낮은 신뢰도 기사 재검토 가능
- 카테고리별 기사 수 조회 가능
- 테스트 코드와 샘플 뉴스 케이스 추가
