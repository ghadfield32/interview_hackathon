# ─────────────────────────────────────────────────────────────────────────────
# Load RAILWAY_API_TOKEN from root .env (one level up from this scripts/ folder)
# ─────────────────────────────────────────────────────────────────────────────
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$envFile   = Join-Path $scriptDir '..\.env'

if (Test-Path $envFile) {
    Write-Host "Reading .env from $envFile"
    Get-Content $envFile | ForEach-Object {
        # matches lines like RAILWAY_API_TOKEN=abc or RAILWAY_API_TOKEN="abc"
        if ($_ -match '^\s*RAILWAY_API_TOKEN\s*=\s*"?(.+?)"?\s*$') {
            $token = $Matches[1]
            $Env:RAILWAY_API_TOKEN = $token
            Write-Host "Loaded RAILWAY_API_TOKEN from .env"
            return
        }
    }
    if (-not $Env:RAILWAY_API_TOKEN) {
        Write-Warning "RAILWAY_API_TOKEN not found in .env. You may need to add it."
    }
} else {
    Write-Warning ".env file not found at $envFile"
}

# ─────────────────────────────────────────────────────────────────────────────
# Clear existing Railway CLI configuration
# ─────────────────────────────────────────────────────────────────────────────
Remove-Item -Force "$Env:USERPROFILE\.railway\config.json" -ErrorAction SilentlyContinue
Test-Path "$Env:USERPROFILE\.railway\config.json"  # Should return False

# ─────────────────────────────────────────────────────────────────────────────
# Clear any old env vars
# ─────────────────────────────────────────────────────────────────────────────
Remove-Item Env:RAILWAY_TOKEN      -ErrorAction SilentlyContinue
Remove-Item Env:RAILWAY_API_TOKEN  -ErrorAction SilentlyContinue

# ─────────────────────────────────────────────────────────────────────────────
# Re-set from .env (in case Remove-Item wiped it)
# ─────────────────────────────────────────────────────────────────────────────
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*RAILWAY_API_TOKEN\s*=\s*"?(.+?)"?\s*$') {
            $Env:RAILWAY_API_TOKEN = $Matches[1]
        }
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Railway logout/login using the loaded token
# ─────────────────────────────────────────────────────────────────────────────
railway logout
railway whoami  # should show “not logged in”
railway login  # non-interactively picks up $Env:RAILWAY_API_TOKEN
railway whoami  # verify you’re back in
