$ErrorActionPreference = "Stop"

$env:PYTHONPATH = "src"

& "..\.venv\Scripts\python.exe" -m news_classifier.cli collect `
    --limit 10 `
    --csv out.csv `
    --sqlite news.db

exit $LASTEXITCODE
