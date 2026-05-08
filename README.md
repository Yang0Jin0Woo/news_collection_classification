# 뉴스 수집 및 분류기

Python 기반 뉴스 수집 및 분류기 프로젝트입니다.

Google News RSS에서 키워드 기반 뉴스를 수집하고, Hugging Face zero-shot 분류 모델과 규칙 기반 보정 로직을 활용하여 뉴스 카테고리를 분류합니다.  
분류 결과는 CSV 파일로 저장할 수 있으며, SQLite DB에 누적 저장하여 필요할 때 다시 조회할 수 있습니다.

![alt text](image.png)

![alt text](image-1.png)

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

## 프로젝트 구조

```text
news_classifier_expanded/
├─ src/
│  └─ news_classifier/
│     ├─ cli.py
│     ├─ pipeline.py
│     ├─ collectors/
│     ├─ classifiers/
│     ├─ dedup/
│     ├─ exporters/
│     ├─ reporting/
│     ├─ storage/
│     └─ utils/
├─ tests/
├─ data/
├─ scripts/
├─ README.md
├─ requirements.txt
├─ pyproject.toml
└─ .gitignore
```

---

## 실행 환경

- Python 3.12 이상 권장
- Windows PowerShell 기준
- VS Code 터미널 기준

---

## 1. 프로젝트 폴더로 이동

아래 명령어를 PowerShell 터미널에 입력합니다.

```powershell
cd C:\Users\juyan\Downloads\news_classifier_expanded_project\news_classifier_expanded
```

---

## 2. 가상환경 생성

```powershell
py -m venv .venv
```

---

## 3. 가상환경 활성화

PowerShell에서 아래 명령어를 실행합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

정상적으로 활성화되면 터미널 앞에 `(.venv)`가 표시됩니다.

```powershell
(.venv) PS C:\Users\juyan\Downloads\news_classifier_expanded_project\news_classifier_expanded>
```

---

## 4. 패키지 설치

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 5. PYTHONPATH 설정

`src` 폴더 안의 패키지를 인식하도록 환경변수를 설정합니다.

```powershell
$env:PYTHONPATH="src"
```

---

## 6. CSV 파일로 결과 저장

뉴스 수집 및 분류 결과를 `out.csv`로 저장합니다.

```powershell
python -m news_classifier.cli collect --keyword "AI 반도체" --limit 10 --csv out.csv
```

실행 후 현재 폴더에 아래 파일이 생성됩니다.

```text
out.csv
```

CSV 파일은 Excel로 열어 결과를 확인할 수 있습니다.

```powershell
ii .\out.csv
```

또는 현재 폴더를 파일 탐색기로 열어 확인할 수 있습니다.

```powershell
explorer .
```

---

## 7. CSV와 SQLite에 동시에 저장

뉴스 결과를 CSV로 확인하면서 SQLite DB에도 누적 저장하려면 아래 명령어를 사용합니다.

```powershell
python -m news_classifier.cli collect --keyword "AI 반도체" --limit 10 --csv out.csv --sqlite news.db
```

실행 후 생성되는 파일은 다음과 같습니다.

```text
out.csv
news.db
```

| 파일 | 용도 |
|---|---|
| `out.csv` | Excel로 결과 확인 |
| `news.db` | 뉴스 분류 결과 누적 저장 및 재조회 |

---

## 8. 다른 키워드로 실행하기

키워드를 바꾸면 다른 주제의 뉴스도 수집할 수 있습니다.

```powershell
python -m news_classifier.cli collect --keyword "생성형 AI" --limit 10 --csv generative_ai.csv --sqlite news.db
```

```powershell
python -m news_classifier.cli collect --keyword "반도체 공급망" --limit 10 --csv semiconductor_supply.csv --sqlite news.db
```

SQLite 파일명을 동일하게 `news.db`로 지정하면 여러 키워드의 결과를 하나의 DB에 누적 저장할 수 있습니다.

---

## 9. SQLite 저장 결과 확인

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

## 10. 테스트 실행

프로젝트 테스트 코드를 실행합니다.

```powershell
python -m pytest
```

---

## 실행 예시

아래는 처음부터 실행할 때 사용하는 전체 명령어 예시입니다.

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

---

## 결과 파일 관리

아래 파일들은 실행 결과물이므로 GitHub에는 올리지 않습니다.

```text
out.csv
news.db
*.sqlite
*.sqlite3
.venv/
__pycache__/
.pytest_cache/
```

실행 결과를 포트폴리오에 첨부하고 싶다면 `sample_result.csv`처럼 별도 샘플 파일명으로 저장해 관리하는 것을 권장합니다.

---

## GitHub 업로드 예시

프로젝트를 GitHub에 업로드할 때는 아래 명령어를 사용할 수 있습니다.

```powershell
git init
git add .
git status
git commit -m "뉴스 수집 및 분류기 프로젝트 초기 커밋"
git branch -M main
git remote add origin https://github.com/Yang0Jin0Woo/news-collection-classification_python.git
git push -u origin main
```

이미 `origin`이 존재한다는 오류가 나오면 아래 명령어를 사용합니다.

```powershell
git remote set-url origin https://github.com/Yang0Jin0Woo/news-collection-classification_python.git
git push -u origin main
```

---

## 프로젝트 개선 내용

초기 버전은 단일 Python 스크립트에서 뉴스 수집, 분류, 규칙 보정, CSV 저장을 처리하는 구조였습니다.

이후 기능을 분리하고 SQLite 저장 기능을 추가하여 단발성 CSV 확인뿐 아니라 뉴스 분류 결과를 누적 저장하고 다시 조회할 수 있도록 개선했습니다.

주요 개선 사항은 다음과 같습니다.

- CSV 저장과 SQLite 누적 저장 기능 분리
- 키워드별 뉴스 수집 결과 누적 관리
- 최종 카테고리, 신뢰도, 규칙 보정 여부 저장
- 낮은 신뢰도 기사 재검토 가능
- 카테고리별 기사 수 조회 가능
- 모델 결과와 규칙 기반 보정 결과를 함께 저장하여 분류 근거 확인 가능

---

## 포트폴리오 설명 문구

Python 기반 뉴스 수집 및 분류기 프로젝트로, Google News RSS에서 키워드 기반 뉴스를 수집하고 Hugging Face zero-shot 분류 모델을 활용해 뉴스 주제를 분류했습니다.

모델 결과가 애매한 경우에는 모델 점수와 top1-top2 margin을 기준으로 판단하고, 규칙 기반 키워드 보정을 적용해 최종 카테고리와 신뢰도를 산출했습니다.

또한 CSV 저장 기능과 SQLite 누적 저장 기능을 분리하여, 실행 결과는 Excel로 확인하고 과거 분류 결과는 DB에서 다시 조회할 수 있도록 개선했습니다.