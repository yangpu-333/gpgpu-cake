param(
    [Parameter(Mandatory=$true)][string]$PromptFile,
    [Parameter(Mandatory=$true)][string]$OutputFile,
    [int]$MaxTurns = 8,
    [string]$ResumeSession
)
$ErrorActionPreference = 'Stop'
$checkout = Join-Path (Split-Path $PSScriptRoot) 'upstream-kda'
$key = $env:PARATERA_API_KEY
if ([string]::IsNullOrWhiteSpace($key)) { $key = [Environment]::GetEnvironmentVariable('PARATERA_API_KEY','User') }
if ([string]::IsNullOrWhiteSpace($key)) { throw 'Configured Paratera key unavailable.' }
if (Test-Path -LiteralPath $OutputFile) { throw 'Immutable CLI evidence already exists.' }
$prompt = Get-Content -Raw -LiteralPath $PromptFile
$oldBase = $env:ANTHROPIC_BASE_URL
$oldToken = $env:ANTHROPIC_AUTH_TOKEN
$oldApiKey = $env:ANTHROPIC_API_KEY
$oldOutputEncoding = $OutputEncoding
$OutputEncoding = New-Object Text.UTF8Encoding($false)
$env:ANTHROPIC_BASE_URL = 'https://llmapi.paratera.com'
$env:ANTHROPIC_AUTH_TOKEN = $key
Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
try {
    Push-Location $checkout
    try {
        $cliArgs = @('-p','--model','Claude-Opus-4.8','--max-turns',"$MaxTurns",
            '--tools','Read','--allowedTools','Read','--add-dir',$PSScriptRoot,
            '--output-format','json','--setting-sources','project,local')
        if ($ResumeSession) { $cliArgs += @('--resume',$ResumeSession) }
        $raw = $prompt | & claude @cliArgs 2>&1
        $exitCode = $LASTEXITCODE
    } finally { Pop-Location }
    $text = ($raw | Out-String).Replace($key,'[REDACTED]')
    New-Item -ItemType Directory -Path (Split-Path $OutputFile) -Force | Out-Null
    [IO.File]::WriteAllText($OutputFile,$text,(New-Object Text.UTF8Encoding($false)))
    $response = $text | ConvertFrom-Json
    [pscustomobject]@{Evidence=$OutputFile;ExitCode=$exitCode;IsError=$response.is_error;Turns=$response.num_turns;
        Model=($response.modelUsage.PSObject.Properties.Name -join ',')} | ConvertTo-Json -Compress
    if ($exitCode -ne 0 -or $response.is_error) { exit 1 }
} finally {
    if ($null -eq $oldBase) { Remove-Item Env:ANTHROPIC_BASE_URL -ErrorAction SilentlyContinue } else { $env:ANTHROPIC_BASE_URL=$oldBase }
    if ($null -eq $oldToken) { Remove-Item Env:ANTHROPIC_AUTH_TOKEN -ErrorAction SilentlyContinue } else { $env:ANTHROPIC_AUTH_TOKEN=$oldToken }
    if ($null -eq $oldApiKey) { Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue } else { $env:ANTHROPIC_API_KEY=$oldApiKey }
    $OutputEncoding = $oldOutputEncoding
    $key=$null
}
