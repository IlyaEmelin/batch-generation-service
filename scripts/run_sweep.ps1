# Замер concurrency на фиксированной подвыборке benchmark-набора.
#
# Каждый прогон пишет outputs/sweep_c<N>.jsonl и рядом сводку
# outputs/sweep_c<N>.summary.json; из этих сводок check_metrics.py собирает
# таблицу concurrency в report.json.
param(
    [string]$Concurrency = "1,4,16,32",
    [int]$Rows = 24,
    [string]$InputFile = "benchmark.jsonl",
    [string]$Python = "C:\Users\Илья\PyCharmMiscProject\.venv\Scripts\python.exe"
)

# Список уровней принимаем строкой: при запуске через -File список с запятыми
# разбирается на отдельные аргументы, а `1;4;16` доезжает одним токеном.
$levels = $Concurrency -split '[;,]' | ForEach-Object { $_.Trim() } | Where-Object { $_ } | ForEach-Object { [int]$_ }

foreach ($level in $levels) {
    Write-Host "=== concurrency=$level ==="
    & $Python run_batch.py --input $InputFile --rows $Rows --concurrency $level `
        --output "outputs/sweep_c$level.jsonl" 2>&1 | Tee-Object -FilePath "outputs/sweep_c$level.log"
}
