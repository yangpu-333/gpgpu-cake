param(
    [Parameter(Mandatory=$true)][string]$PromptFile,
    [Parameter(Mandatory=$true)][string]$OutputFile,
    [int]$MaxTurns = 8
)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$checkout = Join-Path (Split-Path $taskRoot) 'upstream-kda'
$key = $env:PARATERA_API_KEY
if ([string]::IsNullOrWhiteSpace($key)) { $key = [Environment]::GetEnvironmentVariable('PARATERA_API_KEY', 'User') }
if ([string]::IsNullOrWhiteSpace($key)) { throw 'Configured Paratera key is unavailable.' }
if (Test-Path -LiteralPath $OutputFile) { throw 'Refusing to overwrite CLI evidence.' }
$prompt = Get-Content -Raw -LiteralPath $PromptFile
$env:ANTHROPIC_BASE_URL = 'https://llmapi.paratera.com'
$env:ANTHROPIC_AUTH_TOKEN = $key
Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
Push-Location $checkout
try {
    $raw = & claude -p $prompt --model 'Claude-Opus-4.8' --max-turns $MaxTurns `
        --tools Read --allowedTools Read --add-dir $taskRoot --output-format json --setting-sources 'project,local' 2>&1
    $exitCode = $LASTEXITCODE
} finally { Pop-Location }
$rawText = ($raw | Out-String).Replace($key, '[REDACTED]')
$key = $null
$parent = Split-Path $OutputFile
New-Item -ItemType Directory -Path $parent -Force | Out-Null
[IO.File]::WriteAllText($OutputFile, $rawText, (New-Object Text.UTF8Encoding($false)))
$response = $rawText | ConvertFrom-Json
[pscustomobject]@{Evidence=$OutputFile;ExitCode=$exitCode;IsError=$response.is_error;
    Turns=$response.num_turns;Model=($response.modelUsage.PSObject.Properties.Name -join ',')} | ConvertTo-Json -Compress
if ($exitCode -ne 0 -or $response.is_error) { exit 1 }
