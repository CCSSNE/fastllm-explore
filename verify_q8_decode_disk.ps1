param(
    [int]$MaxNewTokens = 6,
    [int]$Threads = 12,
    [int]$DiskMoeLoadThreads = 16,
    [string]$Prompt = "",
    [switch]$Profile
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$dataDirName = -join ([char]0x8D44, [char]0x6599)
$resultDirName = -join ([char]0x9A8C, [char]0x8BC1, [char]0x7ED3, [char]0x679C)
$resultDir = Join-Path (Join-Path $root $dataDirName) $resultDirName
New-Item -ItemType Directory -Force -Path $resultDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $resultDir "q8_long_decode_$stamp.log"
$csvPath = Join-Path $resultDir "q8_long_decode_disk_$stamp.csv"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:DSV4_THREADS = [string]$Threads
$env:DSV4_MAX_NEW_TOKENS = [string]$MaxNewTokens
$env:FASTLLM_DISK_MOE_LOAD_THREADS = [string]$DiskMoeLoadThreads
if ($Profile) {
    $env:FASTLLM_DSV4_PROFILE = "1"
} else {
    Remove-Item Env:FASTLLM_DSV4_PROFILE -ErrorAction SilentlyContinue
}

Set-Content -Encoding UTF8 -Path $csvPath -Value "timestamp,d_read_MBps,d_reads_per_sec,d_avg_read_ms,d_queue,total_read_MBps"

$counterJob = Start-Job -ArgumentList $csvPath -ScriptBlock {
    param($csv)
    while ($true) {
        $sample = Get-Counter -Counter "\PhysicalDisk(1 D:)\Disk Read Bytes/sec","\PhysicalDisk(1 D:)\Disk Reads/sec","\PhysicalDisk(1 D:)\Avg. Disk sec/Read","\PhysicalDisk(1 D:)\Current Disk Queue Length","\PhysicalDisk(_Total)\Disk Read Bytes/sec"
        $values = @{}
        foreach ($counter in $sample.CounterSamples) {
            $values[$counter.Path] = $counter.CookedValue
        }
        $dRead = $values["\\$env:COMPUTERNAME\physicaldisk(1 d:)\disk read bytes/sec"] / 1MB
        $dReads = $values["\\$env:COMPUTERNAME\physicaldisk(1 d:)\disk reads/sec"]
        $dAvgMs = $values["\\$env:COMPUTERNAME\physicaldisk(1 d:)\avg. disk sec/read"] * 1000
        $dQueue = $values["\\$env:COMPUTERNAME\physicaldisk(1 d:)\current disk queue length"]
        $totalRead = $values["\\$env:COMPUTERNAME\physicaldisk(_total)\disk read bytes/sec"] / 1MB
        Add-Content -Encoding UTF8 -Path $csv -Value ("{0},{1:F2},{2:F2},{3:F3},{4:F2},{5:F2}" -f (Get-Date).ToString("o"), $dRead, $dReads, $dAvgMs, $dQueue, $totalRead)
        Start-Sleep -Seconds 2
    }
}

try {
    Push-Location $root
    if ([string]::IsNullOrWhiteSpace($Prompt)) {
        python .\verify_q8_decode_tokens.py 2>&1 | Tee-Object -FilePath $logPath
    } else {
        python .\verify_q8_decode_tokens.py $Prompt 2>&1 | Tee-Object -FilePath $logPath
    }
}
finally {
    Pop-Location
    Stop-Job $counterJob -ErrorAction SilentlyContinue | Out-Null
    Remove-Job $counterJob -Force -ErrorAction SilentlyContinue | Out-Null
}

$rows = Import-Csv -Path $csvPath
$readValues = $rows | ForEach-Object { [double]$_.d_read_MBps }
$lastValues = $readValues | Select-Object -Last 30

$events = Select-String -Path $logPath -Pattern "TOKEN_EVENT|done_sec=|loaded_sec=" | ForEach-Object { $_.Line }

[pscustomobject]@{
    Log = $logPath
    DiskCsv = $csvPath
    Samples = $readValues.Count
    AvgReadMBps = [math]::Round(($readValues | Measure-Object -Average).Average, 1)
    MaxReadMBps = [math]::Round(($readValues | Measure-Object -Maximum).Maximum, 1)
    Last30AvgReadMBps = [math]::Round(($lastValues | Measure-Object -Average).Average, 1)
    Last30MaxReadMBps = [math]::Round(($lastValues | Measure-Object -Maximum).Maximum, 1)
    Events = $events -join " | "
} | Format-List
