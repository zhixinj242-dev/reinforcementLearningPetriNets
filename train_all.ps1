# Windows PowerShell 批量训练脚本
# 对应 train_all.sh 的功能，但适配 Windows 环境

$exploration_timesteps = 80000
$exploration_final_epsilon = 0.04
$learning_starts = 10000
$random_timesteps = 10000

# 8组参数组合 (对应用户提供的 params)
# 格式字符串: "m_success m_cars_driven m_waiting_time m_max_waiting_time m_timestep"
$params = @(
    "0.0 0.0 1.0 0.0 0.0",
    "0.0 0.0 1.0 1.5 0.0",
    "1.0 0.0 1.0 0.0 0.0",
    "1.0 0.0 1.0 1.5 0.0",
    "1.5 0.0 1.0 0.0 0.0",
    "1.5 0.0 1.0 1.5 0.0",
    "2.0 0.0 1.0 0.0 0.0",
    "2.0 0.0 1.0 1.5 0.0"
)

$total_jobs = $params.Count * 2
$current_job = 0

Write-Host "=========================================="
Write-Host "  开始批量训练 (共 $total_jobs 个任务)"
Write-Host "  注意：本脚本将串行执行所有任务，以避免卡死计算机。"
Write-Host "=========================================="

foreach ($p in $params) {
    # 解析参数 (类似 read -r ms mc mw mmw mt <<< "$param")
    $parts = $p -split " "
    $ms = $parts[0]
    $mc = $parts[1]
    $mw = $parts[2]
    $mmw = $parts[3]
    $mt = $parts[4]

    # --- 任务 1: CDQN ---
    $current_job++
    Write-Host "`n[进度 $current_job / $total_jobs] 正在运行 CDQN (s=$ms, w=$mw, mw=$mmw)..."
    python train.py --train `
        --exploration-timesteps $exploration_timesteps --exploration-final-epsilon $exploration_final_epsilon `
        --learning-starts $learning_starts --random-timesteps $random_timesteps `
        --reward-function dynamic_reward `
        --m-success $ms --m-cars-driven $mc --m-waiting-time $mw `
        --m-max-waiting-time $mmw --m-timestep $mt `
        --constrained

    # --- 任务 2: DQN ---
    $current_job++
    Write-Host "`n[进度 $current_job / $total_jobs] 正在运行 DQN (s=$ms, w=$mw, mw=$mmw)..."
    python train.py --train `
        --exploration-timesteps $exploration_timesteps --exploration-final-epsilon $exploration_final_epsilon `
        --learning-starts $learning_starts --random-timesteps $random_timesteps `
        --reward-function dynamic_reward `
        --m-success $ms --m-cars-driven $mc --m-waiting-time $mw `
        --m-max-waiting-time $mmw --m-timestep $mt `
        --no-constrained
}

Write-Host "`n=========================================="
Write-Host "  所有训练任务已完成！"
Write-Host "=========================================="
