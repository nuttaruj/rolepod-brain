# rolepod-brain, for Windows.
#
# The shell installer is the one the README leads with and the one every other
# platform uses. This is its counterpart, not a port of it: PowerShell is what
# a Windows machine has without asking, and a Windows user should not have to
# install a second shell to install a memory.
#
#   irm https://raw.githubusercontent.com/nuttaruj/rolepod-brain/main/bootstrap.ps1 | iex
#
# It does the same three things in the same order - place a checksum-verified
# binary, fetch the embedding model, wire the CLIs found here - and refuses the
# same things, for the same reasons. Nothing is installed unverified.
#
# What it deliberately does not do is grant anything on the user's behalf.
# Codex will ask its own owner to trust these hooks; that approval is theirs to
# give and this script never writes it.

[CmdletBinding()]
param(
    # Wire one CLI rather than every one found here.
    [string] $Target,
    # Place the binary and stop, changing no configuration.
    [switch] $BinaryOnly,
    # Fetch the embedding model into an existing install and stop.
    [switch] $ModelOnly,
    # A specific release rather than the newest.
    [string] $Version = $env:BRAIN_VERSION,
    # Where the assets come from. Only useful for testing an unpublished build.
    [string] $Base = $env:BRAIN_BASE
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # a progress bar makes a 122 MB download slower

$Repo = 'nuttaruj/rolepod-brain'
$BinDir = if ($env:BRAIN_BIN_DIR) { $env:BRAIN_BIN_DIR } else { Join-Path $HOME '.local\bin' }
$Bin = Join-Path $BinDir 'brain.exe'

function Say($message) { Write-Host $message }
function Die($message) { Write-Error $message; exit 1 }

# One build, deliberately. Only x64 is published, and Windows on ARM runs an
# x64 binary through emulation - slower than a native build would be, and the
# alternative for those machines is nothing at all. When there is an arm64
# build to give them, this is where the choice goes.
$platform = 'x86_64-pc-windows-msvc'

if (-not $Version) {
    $Version = (Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest").tag_name
}
if (-not $Base) { $Base = "https://github.com/$Repo/releases/download/$Version" }

# Nothing is installed without matching what the release says it should be.
# This binary reads what you type into your editor.
function Get-Verified($url, $name, $sums, $into) {
    Invoke-WebRequest -Uri $url -OutFile $into
    $want = (Select-String -Path $sums -Pattern "\s$([regex]::Escape($name))$" |
        Select-Object -First 1).Line -split '\s+' | Select-Object -First 1
    if (-not $want) { Die "no checksum published for $name - refusing to install" }
    $got = (Get-FileHash -Algorithm SHA256 -Path $into).Hash.ToLower()
    if ($want.ToLower() -ne $got) { Die "checksum mismatch for $name (expected $want, got $got)" }
}

$work = Join-Path ([System.IO.Path]::GetTempPath()) ("brain-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $work -Force | Out-Null
try {
    $sums = Join-Path $work 'SHA256SUMS'
    Invoke-WebRequest -Uri "$Base/SHA256SUMS" -OutFile $sums

    if (-not $ModelOnly) {
        $name = "brain-$platform.exe"
        Say "Fetching $name $Version ..."
        $downloaded = Join-Path $work $name
        Get-Verified "$Base/$name" $name $sums $downloaded
        New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
        # Replaced rather than written over: a running binary cannot be
        # overwritten on Windows, and the error for that is not obvious.
        Remove-Item -Path $Bin -Force -ErrorAction SilentlyContinue
        Move-Item -Path $downloaded -Destination $Bin -Force
        Say "Installed $(& $Bin --version) to $Bin"
    }

    if (-not $BinaryOnly -or $ModelOnly) {
        # Semantic search needs the model; everything else does not. A failure
        # here is reported and stepped over - keyword, entity, neighbour and
        # substring recall all work without it, and `brain doctor` says what
        # that costs.
        $modelDir = & $Bin where --models 2>$null
        if ($modelDir) {
            $weights = Join-Path $modelDir 'model-int8.safetensors'
            if (-not (Test-Path $weights)) {
                Say 'Fetching the embedding model (122 MB, once) ...'
                New-Item -ItemType Directory -Path $modelDir -Force | Out-Null
                try {
                    foreach ($file in 'model-int8.safetensors', 'tokenizer.json') {
                        $into = Join-Path $work $file
                        Get-Verified "$Base/$file" $file $sums $into
                    }
                    # Renamed into place only once both are whole, so a broken
                    # download is never a half-installed model that loads and
                    # answers differently.
                    foreach ($file in 'model-int8.safetensors', 'tokenizer.json') {
                        Move-Item (Join-Path $work $file) (Join-Path $modelDir $file) -Force
                    }
                    Say 'Semantic search is ready.'
                } catch {
                    Say "Could not fetch the embedding model; run 'brain doctor' for what that costs."
                }
            }
        }
    }
} finally {
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}

if ($ModelOnly) { exit 0 }

if ($BinDir -notin ($env:PATH -split ';')) {
    Say ''
    Say "NOTE: $BinDir is not on your PATH. Add it for this user with:"
    Say "  [Environment]::SetEnvironmentVariable('PATH', `"`$env:PATH;$BinDir`", 'User')"
}

if ($BinaryOnly) {
    Say ''
    Say "Skipped hook registration. Run '$Bin setup' to see what it would change."
    exit 0
}

# No target means every supported CLI found here, which is what someone piping
# this into a shell almost always wants.
Say ''
Say 'Planned changes:'
$plan = if ($Target) { @('setup', '--cli', $Target) } else { @('setup') }
& $Bin @plan
Say ''
$reply = Read-Host 'Apply these changes? [y/N]'
if ($reply -match '^[Yy]') {
    & $Bin @plan --apply
} else {
    Say "Nothing changed. Run '$Bin setup --apply' when you want to."
}
