# Python News Collector & Classifier

신입 개발자 포트폴리오용으로 구성한 **뉴스 수집 및 분류 자동화 프로젝트 확장판**입니다.
기존 단일 파일 파이프라인을 다음과 같이 실무형 구조로 분리했습니다.

- Google News RSS 수집
- 기사 본문 추출 및 정제
- 중복 기사 제거
- Hugging Face zero-shot 분류
- 규칙 기반 보정
- 신뢰도/마진 판단
- SQLite/CSV 저장
- Excel 리포트 출력
- CLI 실행
- Streamlit 대시보드 예시
- 테스트 코드와 샘플 케이스

## 핵심 설계 의도

이 프로젝트는 단순히 줄 수를 늘리기 위한 코드가 아니라, 면접에서 설명 가능한 구조를 목표로 합니다.

1. 수집 계층과 분류 계층을 분리했습니다.
2. 모델 결과를 그대로 쓰지 않고 confidence, margin, rule score를 함께 봅니다.
3. 규칙 기반 보정은 근거를 남기도록 설계했습니다.
4. 저장소 계층을 CSV와 SQLite로 분리했습니다.
5. 테스트 데이터를 통해 분류 로직의 회귀를 확인할 수 있게 했습니다.

## 실행 예시

```bash
python -m news_classifier.cli collect --keyword "AI 반도체" --limit 10 --csv out.csv
python -m news_classifier.cli classify --keyword "전력반도체" --limit 10 --sqlite news.db
```

## 면접 설명용 요약

> 기존에는 하나의 Python 파일에 뉴스 수집, 분류, 규칙 보정, CSV 저장 로직을 모두 작성했습니다. 이후 실무형 구조로 확장하면서 수집기, 분류기, 규칙 엔진, 저장소, 리포트, 테스트 코드를 분리했습니다. 특히 모델 점수만 보지 않고 top1-top2 margin과 키워드 기반 규칙 근거를 함께 판단해 오분류를 줄이는 방향으로 개선했습니다.

## 폴더 구조

```text
src/news_classifier/
  collectors/      # 뉴스 RSS 수집 및 기사 본문 추출
  classifiers/     # zero-shot 모델, 규칙 엔진, 후처리
  dedup/           # 중복 제거
  features/        # 텍스트 특징 추출
  storage/         # CSV, SQLite 저장소
  exporters/       # Excel 리포트
  reporting/       # 요약 리포트 생성
  rules/           # 라벨별 키워드/규칙 사전
  utils/           # 문자열, 시간, HTTP 유틸
  pipeline.py      # 전체 파이프라인 조립
  cli.py           # 명령행 실행 진입점
```

## 포트폴리오에 적을 때

- 단일 스크립트에서 모듈형 프로젝트로 리팩터링
- 모델 예측값, confidence, margin, 규칙 매칭 수를 함께 사용한 후처리
- 기사 제목/설명 정제, 중복 제거, CSV/SQLite 저장 구조 구현
- 샘플 테스트 케이스를 통한 분류 규칙 회귀 검증

## 주의

실제 면접에서는 “몇 줄”보다 “왜 이렇게 나누었는지”가 더 중요합니다. 따라서 줄 수는 전체 저장소 기준으로 말하고, 핵심 파일 규모와 테스트/샘플 데이터 규모를 구분해서 말하는 것이 안전합니다.
