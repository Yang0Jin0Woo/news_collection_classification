param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArguments
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = Split-Path -Parent $projectRoot
$pythonPath = Join-Path $workspaceRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "상위 가상환경의 Python을 찾을 수 없습니다: $pythonPath. README의 실행 방법을 확인하세요."
}

Push-Location $projectRoot
try {
    if ($CliArguments.Count -gt 0) {
        & $pythonPath -m news_classifier.cli @CliArguments
    }
    else {
        # 인수 없는 기존 실행 방식은 유지, Python은 상위 가상환경으로 고정.
        & $pythonPath -m news_classifier.cli collect `
            --limit 10 `
            --csv out.csv `
            --sqlite news.db
    }
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

exit $exitCode
