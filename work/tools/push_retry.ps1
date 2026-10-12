# Retry a kaggle kernels push for a short while when the account's GPU session limit is reached.
# Usage: powershell -File tools\push_retry.ps1 -Dir kernels\wear26-gpjw -Minutes 45 -Accelerator NvidiaTeslaT4
param([string]$Dir, [int]$Minutes = 45, [string]$Accelerator = "NvidiaTeslaT4", [int]$Interval = 300)
$env:PYTHONUTF8 = '1'; $env:PYTHONWARNINGS = 'ignore'
$deadline = (Get-Date).AddMinutes($Minutes)
while ($true) {
    $out = kaggle kernels push -p $Dir --accelerator $Accelerator 2>&1
    $last = ($out | Select-Object -Last 1)
    Write-Output ("{0} {1}" -f (Get-Date).ToString("HH:mm"), $last)
    if ($out -match 'successfully') { exit 0 }
    if (-not ($out -match 'Maximum batch GPU session')) { exit 2 }
    if ((Get-Date) -gt $deadline) { Write-Output "giving up for now (GPU sessions busy)"; exit 3 }
    Start-Sleep -Seconds $Interval
}
