#Requires -Version 5.1
param(
    [switch]$NoGui
)

$ErrorActionPreference = 'Stop'

# Auto-elevate to Administrator if needed
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Requesting Administrator privileges..." -ForegroundColor Yellow
    $argList = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    if ($NoGui) { $argList += " -NoGui" }
    Start-Process powershell.exe -Verb RunAs -ArgumentList $argList
    exit 0
}

$RepoRoot  = Split-Path -Parent $PSScriptRoot
$DataRoot  = 'C:\ProgramData\FaceUnlock'
$BinDir    = Join-Path $DataRoot 'bin'
$ModelsDir = Join-Path $DataRoot 'models'
$VaultDir  = Join-Path $DataRoot 'vault'

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Windows 11 Face ID (C++20 Zero-Daemon + 120-165Hz HUD)  " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

Write-Host "[1/5] Provisioning system directories in $DataRoot..." -ForegroundColor White
New-Item -ItemType Directory -Force -Path $BinDir, $ModelsDir, $VaultDir | Out-Null

# Lock Vault ACL strictly to SYSTEM and Administrators
& icacls.exe $VaultDir /inheritance:r /grant:r "NT AUTHORITY\SYSTEM:(OI)(CI)(F)" "BUILTIN\Administrators:(OI)(CI)(F)" | Out-Null

Write-Host "[2/5] Installing C++20 Biometric Engine & ONNX Runtime..." -ForegroundColor White
$RequiredDist = @(
    @{ Src = (Join-Path $RepoRoot 'dist\FaceCredentialProvider.dll');      Dst = (Join-Path $BinDir 'FaceCredentialProvider.dll') },
    @{ Src = (Join-Path $RepoRoot 'dist\onnxruntime.dll');                 Dst = (Join-Path $BinDir 'onnxruntime.dll') },
    @{ Src = (Join-Path $RepoRoot 'dist\onnxruntime_providers_shared.dll');Dst = (Join-Path $BinDir 'onnxruntime_providers_shared.dll') },
    @{ Src = (Join-Path $RepoRoot 'FaceIdManager.exe');                    Dst = (Join-Path $BinDir 'FaceIdManager.exe') }
)
foreach ($item in $RequiredDist) {
    if (-not (Test-Path $item.Src)) {
        throw "Missing required binary: $($item.Src). Run scripts\build.ps1 first."
    }
    Copy-Item $item.Src $item.Dst -Force
}

Write-Host "[3/5] Installing optimized ONNX models (YuNet, SFace, MiniFASNet)..." -ForegroundColor White
foreach ($m in @('yunet_opt.onnx', 'sface_opt.onnx', 'fas1_opt.onnx', 'fas2_opt.onnx')) {
    $mSrc = Join-Path $RepoRoot "models\$m"
    if (-not (Test-Path $mSrc)) {
        throw "Missing required ONNX model: $mSrc"
    }
    Copy-Item $mSrc (Join-Path $ModelsDir $m) -Force
}

Write-Host "[4/5] Enabling Microsoft Account Password + PIN coexistence..." -ForegroundColor White
$pwLessKey = 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\PasswordLess\Device'
if (-not (Test-Path $pwLessKey)) {
    New-Item -Path $pwLessKey -Force | Out-Null
}
Set-ItemProperty -Path $pwLessKey -Name 'DevicePasswordLessBuildVersion' -Value 0 -Type DWord

Write-Host "[5/5] Registering Windows 11 Lock Screen Credential Provider..." -ForegroundColor White
$SysDll = Join-Path $BinDir 'FaceCredentialProvider.dll'
& regsvr32.exe /s $SysDll
if ($LASTEXITCODE -ne 0) {
    throw "regsvr32 failed to register $SysDll (exit code $LASTEXITCODE)"
}

Write-Host "`nInstallation complete! (0 MB background RAM - 100% On-Demand)" -ForegroundColor Green
Write-Host "Next steps inside Face ID Control Center:" -ForegroundColor Yellow
Write-Host "  1. Click [Quet & Dang ky lai 15 mau mat] to enroll your face (~5s)."
Write-Host "  2. Enter your Windows / Microsoft Account password and click [Kiem tra & Luu]."
Write-Host "  3. Press Win + L to experience instant Face ID unlock!"

if (-not $NoGui) {
    Start-Process (Join-Path $BinDir 'FaceIdManager.exe')
}
