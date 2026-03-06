@echo off
REM 【文件角色】：Windows批处理脚本，用于监控远程训练进度。
REM 使用方法：monitor_training.bat

echo ===== 远程训练监控工具 =====
echo.

REM 设置服务器连接信息（请根据实际情况修改）
set SERVER_IP=your_server_ip
set SERVER_PORT=22
set SERVER_USER=your_username
set PROJECT_PATH=/autodl-tmp/petri RL

REM 检查是否已配置服务器信息
if "%SERVER_IP%"=="your_server_ip" (
    echo 请先编辑此文件，设置正确的服务器连接信息
    echo 需要修改的变量：
    echo   SERVER_IP - 服务器IP地址
    echo   SERVER_PORT - SSH端口（默认22）
    echo   SERVER_USER - 服务器用户名
    echo   PROJECT_PATH - 项目路径
    pause
    exit /b 1
)

echo 服务器信息：
echo   IP地址: %SERVER_IP%
echo   端口: %SERVER_PORT%
echo   用户名: %SERVER_USER%
echo   项目路径: %PROJECT_PATH%
echo.

REM 选择监控模式
echo 请选择监控模式：
echo 1. 单次监控
echo 2. 连续监控（每60秒刷新）
echo 3. 连续监控（自定义间隔）
set /p choice="请输入选择 (1-3): "

if "%choice%"=="1" (
    echo.
    echo 执行单次监控...
    python monitor_remote_training.py --server-ip %SERVER_IP% --server-port %SERVER_PORT% --server-user %SERVER_USER% --project-path "%PROJECT_PATH%"
) else if "%choice%"=="2" (
    echo.
    echo 开始连续监控（每60秒刷新）...
    python monitor_remote_training.py --server-ip %SERVER_IP% --server-port %SERVER_PORT% --server-user %SERVER_USER% --project-path "%PROJECT_PATH%" --continuous
) else if "%choice%"=="3" (
    echo.
    set /p interval="请输入刷新间隔（秒）: "
    echo 开始连续监控（每%interval%秒刷新）...
    python monitor_remote_training.py --server-ip %SERVER_IP% --server-port %SERVER_PORT% --server-user %SERVER_USER% --project-path "%PROJECT_PATH%" --continuous --interval %interval%
) else (
    echo 无效选择，退出
    pause
    exit /b 1
)

echo.
echo 监控完成
pause