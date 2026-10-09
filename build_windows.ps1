$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

py -m PyInstaller --noconfirm --clean --onefile --windowed `
    --name GbaEmulator `
    --distpath (Join-Path $PSScriptRoot "dist") `
    --workpath (Join-Path $PSScriptRoot "build\pyinstaller-work") `
    --specpath (Join-Path $PSScriptRoot "build\pyinstaller") `
    --collect-all PIL `
    (Join-Path $PSScriptRoot "emulator_app.py")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller did not build the Windows app." }

$executable = Join-Path $PSScriptRoot "dist\GbaEmulator.exe"
$readme = Join-Path $PSScriptRoot "dist\README.txt"
$demoSource = Join-Path $PSScriptRoot "examples\gradient_demo.gba"
$demo = Join-Path $PSScriptRoot "dist\gradient_demo.gba"
$archive = Join-Path $PSScriptRoot "dist\GbaEmulator-Windows-x64-with-demo.zip"
if (-not (Test-Path -LiteralPath $executable)) { throw "The executable was not created." }
if (-not (Test-Path -LiteralPath $demoSource)) { throw "The original homebrew demo ROM is missing." }
Copy-Item -LiteralPath $demoSource -Destination $demo -Force

@'
GBA Emulator — Windows x64

Extract the ZIP and open GbaEmulator.exe. Choose gradient_demo.gba with Open ROM…,
then click Run. Python is not required; this build is for 64-bit Windows 10 or newer.

Controls: arrows = D-pad; Z/X = A/B; A/S = L/R; Enter/Backspace = Start/Select.

This early emulator supports a subset of GBA hardware. Some games may pause on
unsupported instructions. No games or Nintendo BIOS are included. Use ROMs you
have the right to use. Saves are written beside the selected ROM.
'@ | Set-Content -LiteralPath $readme -Encoding utf8

Compress-Archive -LiteralPath $executable, $readme, $demo -DestinationPath $archive -Force
Write-Host "Built $executable"
Write-Host "Packaged $archive"
