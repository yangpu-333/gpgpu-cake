param(
    [switch]$ConfigureKey,
    [switch]$CheckOnly,
    [string]$Model = 'Claude-Opus-4.8'
)

$ErrorActionPreference = 'Stop'
$baseUrl = 'https://llmapi.paratera.com'
$checkout = Join-Path $PSScriptRoot 'upstream-kda'
$adapted = Join-Path $env:USERPROFILE '.codex\skills\kernelwiki-iluvatar'

if ($ConfigureKey) {
    $secureKey = Read-Host -AsSecureString 'Paratera API Key'
    $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    try {
        $key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
        if ([string]::IsNullOrWhiteSpace($key)) { throw 'API Key is empty.' }
        [Environment]::SetEnvironmentVariable('PARATERA_API_KEY', $key, 'User')
    } finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
        $key = $null
    }
    Write-Output 'Paratera API Key saved in this Windows user environment. The value was not printed.'
    return
}

if (-not (Test-Path -LiteralPath (Join-Path $checkout 'skills\KernelWiki\SKILL.md'))) {
    throw "KDA checkout is missing: $checkout"
}

& (Join-Path $PSScriptRoot 'link_kda_skills.ps1') `
    -KdaCheckout $checkout `
    -AdaptedSkill $adapted `
    -SkillDirectory (Join-Path $checkout '.claude\skills') | Out-Null

if ($CheckOnly) {
    Write-Output "Claude Code project skills ready in $checkout"
    Write-Output "API base: $baseUrl; model: $Model"
    return
}

$key = $env:PARATERA_API_KEY
if ([string]::IsNullOrWhiteSpace($key)) {
    $key = [Environment]::GetEnvironmentVariable('PARATERA_API_KEY', 'User')
}
if ([string]::IsNullOrWhiteSpace($key)) {
    throw "No Paratera API Key. Run: powershell.exe -NoProfile -File `"$PSCommandPath`" -ConfigureKey"
}
if (-not (Get-Command claude -ErrorAction SilentlyContinue)) {
    throw 'Claude Code CLI is not installed or is not on PATH.'
}

$oldBase = $env:ANTHROPIC_BASE_URL
$oldToken = $env:ANTHROPIC_AUTH_TOKEN
$oldApiKey = $env:ANTHROPIC_API_KEY
$env:ANTHROPIC_BASE_URL = $baseUrl
$env:ANTHROPIC_AUTH_TOKEN = $key
Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue

$prompt = '/kernelwiki-iluvatar Read wiki/hardware/bi-v150-stack.md. Return only the page id, title, confidence, and first source id. Do not edit files.'
$evidenceDir = Join-Path $PSScriptRoot 'evidence'
$output = Join-Path $evidenceDir ('claude-paratera-smoke-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.json')

try {
    Push-Location $checkout
    try {
        $raw = & claude -p $prompt --model $Model --max-turns 3 --tools Read `
            --output-format json --setting-sources 'project,local' 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
    $rawText = $raw | Out-String
    Set-Content -LiteralPath $output -Value $rawText -Encoding utf8
    $response = $rawText | ConvertFrom-Json
    Write-Output "Evidence: $output"
    Write-Output "Exit code: $exitCode; API status: $($response.api_error_status); turns: $($response.num_turns)"
    Write-Output "Result: $($response.result)"
    if ($exitCode -ne 0 -or $response.is_error) { exit 1 }
} finally {
    if ($null -eq $oldBase) { Remove-Item Env:ANTHROPIC_BASE_URL -ErrorAction SilentlyContinue } else { $env:ANTHROPIC_BASE_URL = $oldBase }
    if ($null -eq $oldToken) { Remove-Item Env:ANTHROPIC_AUTH_TOKEN -ErrorAction SilentlyContinue } else { $env:ANTHROPIC_AUTH_TOKEN = $oldToken }
    if ($null -eq $oldApiKey) { Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue } else { $env:ANTHROPIC_API_KEY = $oldApiKey }
}
