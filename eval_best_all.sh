#!/usr/bin/env bash
# 对 16 个实验各跑一次 evaluation.py 选最优，生成 *_best.pt
# 用法：bash eval_best_all.sh

set -e
DIR="lido-run-events"
# 与 evaluation.py 对应：防单 episode 卡死、每 checkpoint 评估轮数
MAX_STEPS=5000
BEST_EPISODES=10

run_eval() {
  python evaluation.py --best-from-dir "$DIR" --exp-name "$1" \
    --max-steps-per-episode "$MAX_STEPS" --best-eval-episodes "$BEST_EPISODES" \
    "${@:2}"
}

run_eval agent_s0.0c0.0w1.0mw0.0t0.0_cdqn
run_eval agent_s0.0c0.0w1.0mw0.0t0.0_dqn --no-constrained
run_eval agent_s0.0c0.0w1.0mw1.5t0.0_cdqn
run_eval agent_s0.0c0.0w1.0mw1.5t0.0_dqn --no-constrained
run_eval agent_s1.0c0.0w1.0mw0.0t0.0_cdqn
run_eval agent_s1.0c0.0w1.0mw0.0t0.0_dqn --no-constrained
run_eval agent_s1.0c0.0w1.0mw1.5t0.0_cdqn
run_eval agent_s1.0c0.0w1.0mw1.5t0.0_dqn --no-constrained
run_eval agent_s1.5c0.0w1.0mw0.0t0.0_cdqn
run_eval agent_s1.5c0.0w1.0mw0.0t0.0_dqn --no-constrained
run_eval agent_s1.5c0.0w1.0mw1.5t0.0_cdqn
run_eval agent_s1.5c0.0w1.0mw1.5t0.0_dqn --no-constrained
run_eval agent_s2.0c0.0w1.0mw0.0t0.0_cdqn
run_eval agent_s2.0c0.0w1.0mw0.0t0.0_dqn --no-constrained
run_eval agent_s2.0c0.0w1.0mw1.5t0.0_cdqn
run_eval agent_s2.0c0.0w1.0mw1.5t0.0_dqn --no-constrained

echo "16 个 *_best.pt 已生成完毕"
