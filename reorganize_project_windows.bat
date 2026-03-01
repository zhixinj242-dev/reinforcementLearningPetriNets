@echo off
echo 开始重构项目目录结构...

REM 1. 创建新的目录结构
echo 创建新的目录结构...
if not exist src mkdir src
if not exist src\agents mkdir src\agents
if not exist src\environment mkdir src\environment
if not exist src\utils mkdir src\utils
if not exist src\rewards mkdir src\rewards

if not exist experiments mkdir experiments
if not exist experiments\configs mkdir experiments\configs
if not exist experiments\results mkdir experiments\results
if not exist experiments\logs mkdir experiments\logs
if not exist experiments\checkpoints mkdir experiments\checkpoints

if not exist models mkdir models
if not exist models\cdqn mkdir models\cdqn
if not exist models\dqn mkdir models\dqn
if not exist models\gail mkdir models\gail

if not exist scripts mkdir scripts
if not exist docs mkdir docs
if not exist data mkdir data

REM 2. 移动源代码
echo 移动源代码...
if exist agents (
    move /Y agents\* src\agents\ 2>nul
    rmdir agents 2>nul
)

if exist environment (
    move /Y environment\* src\environment\ 2>nul
    rmdir environment 2>nul
)

if exist utils (
    if exist utils\petri_net (
        move /Y utils\petri_net\* src\utils\ 2>nul
    )
    move /Y utils\log_manager.py src\utils\ 2>nul
    move /Y utils\result_comparison.py src\utils\ 2>nul
    for %%f in (utils\*.py) do move /Y "%%f" src\utils\ 2>nul
    rmdir utils 2>nul
)

if exist rewards (
    move /Y rewards\* src\rewards\ 2>nul
    rmdir rewards 2>nul
)

REM 3. 整理模型文件
echo 整理模型文件...
if exist lido-run-events (
    REM 移动CDQN模型
    for %%f in (lido-run-events\*_cdqn_*.pt) do move /Y "%%f" models\cdqn\ 2>nul
    for %%f in (lido-run-events\*_cdqn_best.pt) do move /Y "%%f" models\cdqn\ 2>nul
    
    REM 移动DQN模型
    for %%f in (lido-run-events\*_dqn_*.pt) do move /Y "%%f" models\dqn\ 2>nul
    for %%f in (lido-run-events\*_dqn_best.pt) do move /Y "%%f" models\dqn\ 2>nul
    
    REM 移动GAIL模型
    for %%f in (lido-run-events\*gail*.pt) do move /Y "%%f" models\gail\ 2>nul
    
    REM 移动checkpoint文件
    for /d %%d in (lido-run-events\*) do (
        if exist "%%d\checkpoints" (
            xcopy /E /I /Y "%%d\checkpoints\*" experiments\checkpoints\ 2>nul
        )
    )
    
    echo 模型文件整理完成
)

REM 4. 整理日志文件
echo 整理日志文件...
if exist detailed_logs (
    move /Y detailed_logs\* experiments\logs\ 2>nul
    rmdir detailed_logs 2>nul
)

if exist debug_logs (
    move /Y debug_logs\* experiments\logs\ 2>nul
    rmdir debug_logs 2>nul
)

REM 5. 整理结果文件
echo 整理结果文件...
REM 移动CSV结果文件
for %%f in (*.csv) do move /Y "%%f" experiments\results\ 2>nul

REM 移动其他结果文件
for %%f in (*.png) do move /Y "%%f" experiments\results\ 2>nul
for %%f in (*.mp4) do move /Y "%%f" experiments\results\ 2>nul

REM 6. 移动脚本文件
echo 移动脚本文件...
REM 移动shell脚本
for %%f in (*.sh) do move /Y "%%f" scripts\ 2>nul

REM 移动Python脚本
if exist add_to_comparison.py move /Y add_to_comparison.py scripts\ 2>nul
if exist baseline.py move /Y baseline.py scripts\ 2>nul
if exist evaluation.py move /Y evaluation.py scripts\ 2>nul
if exist simulation.py move /Y simulation.py scripts\ 2>nul
if exist train.py move /Y train.py scripts\ 2>nul
if exist visual.py move /Y visual.py scripts\ 2>nul
if exist verify_*.py move /Y verify_*.py scripts\ 2>nul
if exist monitor_*.py move /Y monitor_*.py scripts\ 2>nul
if exist plot_*.py move /Y plot_*.py scripts\ 2>nul
if exist read_*.py move /Y read_*.py scripts\ 2>nul

REM 7. 移动文档文件
echo 移动文档文件...
for %%f in (*.md) do move /Y "%%f" docs\ 2>nul

REM 8. 移动数据文件
echo 移动数据文件...
if exist data (
    echo data目录已存在，保持原样
) else (
    echo 创建data目录
    mkdir data
)

REM 9. 移动plotting目录
if exist plotting (
    move /Y plotting\* scripts\ 2>nul
    rmdir plotting 2>nul
)

REM 10. 移动slurm目录
if exist slurm (
    move /Y slurm\* scripts\ 2>nul
    rmdir slurm 2>nul
)

REM 11. 清理空目录和特殊目录
echo 清理特殊目录...
if exist .ipynb_checkpoints (
    rmdir /S /Q .ipynb_checkpoints
    echo 删除 .ipynb_checkpoints 目录
)

REM 12. 创建新的.gitignore
echo 创建新的.gitignore...
echo # 运行产物 > .gitignore
echo experiments/ >> .gitignore
echo models/ >> .gitignore
echo *.pt >> .gitignore
echo *.csv >> .gitignore
echo *.log >> .gitignore
echo *.mp4 >> .gitignore
echo *.png >> .gitignore
echo. >> .gitignore
echo # Python >> .gitignore
echo __pycache__/ >> .gitignore
echo *.pyc >> .gitignore
echo *.pyo >> .gitignore
echo .venv/ >> .gitignore
echo venv/ >> .gitignore
echo env/ >> .gitignore
echo. >> .gitignore
echo # IDE >> .gitignore
echo .vscode/ >> .gitignore
echo .idea/ >> .gitignore
echo *.swp >> .gitignore
echo *.swo >> .gitignore
echo. >> .gitignore
echo # 系统 >> .gitignore
echo .DS_Store >> .gitignore
echo Thumbs.db >> .gitignore

echo 重构完成！
echo 新的目录结构：
dir /B

echo.
echo === 重构完成后的操作建议 ===
echo 1. 运行 update_paths_windows.py 更新路径引用
echo 2. 测试重构后的功能是否正常
echo 3. 提交重构到版本控制
echo.
echo 主要需要更新的路径：
echo - scripts\train.py 中的日志路径: experiments\logs\
echo - scripts\evaluation.py 中的模型路径: models\
echo - scripts\simulation.py 中的结果路径: experiments\results\
pause