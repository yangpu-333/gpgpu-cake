param(
    [Parameter(Mandatory = $true)]
    [string]$KdaCheckout,

    [Parameter(Mandatory = $true)]
    [string]$AdaptedSkill,

    [string]$SkillDirectory = (Join-Path $env:USERPROFILE '.claude\skills')
)

$ErrorActionPreference = 'Stop'

function Resolve-SkillDirectory([string]$Path) {
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    if (-not (Test-Path -LiteralPath (Join-Path $resolved 'SKILL.md') -PathType Leaf)) {
        throw "Missing SKILL.md: $resolved"
    }
    return $resolved
}

$pinned = Resolve-SkillDirectory (Join-Path $KdaCheckout 'skills\KernelWiki')
$adapted = Resolve-SkillDirectory $AdaptedSkill
New-Item -ItemType Directory -Path $SkillDirectory -Force | Out-Null

foreach ($entry in @(
    @{ Name = 'KernelWiki'; Target = $pinned },
    @{ Name = 'kernelwiki-iluvatar'; Target = $adapted }
)) {
    $link = Join-Path $SkillDirectory $entry.Name
    $existing = Get-Item -LiteralPath $link -Force -ErrorAction SilentlyContinue
    if ($null -ne $existing) {
        if ($existing.LinkType -ne 'Junction' -or
            -not [string]::Equals($existing.Target, $entry.Target, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Different entry already exists: $link"
        }
    } else {
        New-Item -ItemType Junction -Path $link -Target $entry.Target | Out-Null
    }
    Write-Output "$link -> $($entry.Target)"
}
