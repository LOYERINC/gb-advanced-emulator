$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$pythonVersion = py -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')"
$pythonInclude = py -c "import sysconfig; print(sysconfig.get_path('include'))"
$pythonLibDir = py -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
$pythonExtSuffix = py -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
$pythonLibrary = "python$pythonVersion"
$generatedC = Join-Path $PSScriptRoot "build\native\arm7tdmi.c"
$extension = Join-Path $PSScriptRoot "gba\arm7tdmi$pythonExtSuffix"
$nativeExtension = Join-Path $PSScriptRoot "gba\_native_cpu$pythonExtSuffix"

New-Item -ItemType Directory -Force (Split-Path $generatedC) | Out-Null
py -m cython --3str --output-file $generatedC gba\arm7tdmi.py
if ($LASTEXITCODE -ne 0) { throw "Cython could not translate the CPU module." }

py -m ziglang cc -target x86_64-windows-gnu -shared -O2 -U_DEBUG `
    "-I$pythonInclude" "-L$pythonLibDir" "-l$pythonLibrary" `
    -o $extension $generatedC
if ($LASTEXITCODE -ne 0) { throw "The native CPU module did not compile." }

Write-Host "Built $extension"
Write-Host "Python will use the compiled CPU module when it starts."

py -m ziglang cc -target x86_64-windows-gnu -shared -O2 -U_DEBUG `
    "-I$pythonInclude" "-L$pythonLibDir" "-l$pythonLibrary" `
    -o $nativeExtension gba\_native_cpu.c
if ($LASTEXITCODE -ne 0) { throw "The native Thumb runner did not compile." }

Write-Host "Built $nativeExtension"
