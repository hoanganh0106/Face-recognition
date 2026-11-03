#Requires -Version 5.1
param(
    [switch]$PurgeData
)

$ErrorActionPreference = 'Stop'

$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Requesting Administrator privileges..." -ForegroundColor Yellow
    $argList = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    if ($PurgeData) { $argList += " -PurgeData" }
    Start-Process powershell.exe -Verb RunAs -ArgumentList $argList
    exit 0
}

$DataRoot = 'C:\ProgramData\FaceUnlock'
$SysDll   = Join-Path $DataRoot 'bin\FaceCredentialProvider.dll'
$Clsid    = '{F8A0B4D9-3C7F-4B0A-9E21-8C1B1E2B7C10}'

Write-Host "Unregistering Windows 11 Face ID Credential Provider..." -ForegroundColor Cyan
if (Test-Path $SysDll) {
    & regsvr32.exe /u /s $SysDll
}
Remove-Item -Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Authentication\Credential Providers\$Clsid" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "HKLM:\SOFTWARE\Classes\CLSID\$Clsid" -Recurse -Force -ErrorAction SilentlyContinue

if ($PurgeData -and (Test-Path $DataRoot)) {
    Write-Host "Removing $DataRoot (models, enrolled embeddings, DPAPI vault)..." -ForegroundColor Yellow
    Remove-Item -Path $DataRoot -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "Face ID Credential Provider has been cleanly uninstalled." -ForegroundColor Green
