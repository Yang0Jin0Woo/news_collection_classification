$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $projectRoot
try {
    & news-classifier collect `
        --limit 10 `
        --csv out.csv `
        --sqlite news.db
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

exit $exitCode
