# 监控训练进度的 PowerShell 脚本

$LogDir = "train_logs"

if (-not (Test-Path $LogDir)) {
    Write-Host "日志目录 $LogDir 不存在"
    exit 1
}

while ($true) {
    Clear-Host
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host "训练进度监控 - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host ""
    
    $logFiles = Get-ChildItem "$LogDir\*.log" -ErrorAction SilentlyContinue
    
    if ($logFiles.Count -eq 0) {
        Write-Host "没有找到训练日志文件" -ForegroundColor Yellow
    }
    
    foreach ($logFile in $logFiles) {
        $filename = $logFile.Name
        
        # 读取最后 20 行查找进度
        $content = Get-Content $logFile.FullName -Tail 20 -ErrorAction SilentlyContinue
        $progressLine = $content | Select-String -Pattern '\d+%\|.*\|\s*\d+/\d+' | Select-Object -Last 1
        
        if ($progressLine) {
            $progress = $progressLine.Line -replace '.*?(\d+%\|[^|]+\|\s*\d+/\d+).*', '$1'
            Write-Host "[$filename] " -NoNewline -ForegroundColor Green
            Write-Host $progress
        }
        elseif ($content -match "100%") {
            Write-Host "[$filename] ✓ 完成" -ForegroundColor Green
        }
        elseif ($content -match "Error|error|Exception|Traceback") {
            Write-Host "[$filename] ✗ 错误" -ForegroundColor Red
        }
        else {
            Write-Host "[$filename] ... 启动中" -ForegroundColor Yellow
        }
    }
    
    Write-Host ""
    Write-Host "按 Ctrl+C 退出监控" -ForegroundColor Gray
    Start-Sleep -Seconds 2
}
