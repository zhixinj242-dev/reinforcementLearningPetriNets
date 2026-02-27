"""
【文件角色】：批量仿真脚本。
加载由 evaluation.py 选出的最优模型（*_best.pt），对多组参数对应的最优模型进行批量评估，
将生存步数、违规率、等待时间等写入 CSV（simulation-results.csv）。
使用前需先对每组参数运行 evaluation.py --best-from-dir --exp-name 生成 *_best.pt。
"""
import os
import torch
from skrl.agents.torch.dqn import DQN_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env
import pandas as pd
import argparse

from agents.dqn import get_dqn_model
from environment import JunctionPetriNetEnv
from rewards import base_reward
from utils.petri_net import get_petri_net, Parser

# 超参数组合：(s, c, w, mw, t) 等，用于构造模型路径
params = [
    (0.0, 0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 1.5, 0.0),
    (1.0, 0.0, 1.0, 0.0, 0.0),
    (1.0, 0.0, 1.0, 1.5, 0.0),
    (1.5, 0.0, 1.0, 0.0, 0.0),
    (1.5, 0.0, 1.0, 1.5, 0.0),
    (2.0, 0.0, 1.0, 0.0, 0.0),
    (2.0, 0.0, 1.0, 1.5, 0.0)
]

df = pd.DataFrame(columns=['model_type', 's', 'c', 'w', 'mw', 't', 'min_t_frame', 'avg_t_frame', 'max_t_frame', 'min_waiting_time',
                           'avg_waiting_time', 'max_waiting_time', 'min_c_broken', 'avg_c_broken', 'max_c_broken'])


def main(p, model_type="cdqn", render_mode=None, iterations=200):
    """对一组参数对应的最优模型跑多轮评估，并写入 DataFrame 一行。
    
    Args:
        p: (s, c, w, mw, t) 参数元组
        model_type: "cdqn" 或 "dqn"
        render_mode: 渲染模式，None 为无界面（批量评估推荐）
        iterations: 评估轮数
    """
    suffix = "_cdqn" if model_type == "cdqn" else "_dqn"
    exp_name = "agent_s{}c{}w{}mw{}t{}{}".format(p[0], p[1], p[2], p[3], p[4], suffix)
    path = "lido-run-events/{}_best.pt".format(exp_name)
    if not os.path.isfile(path):
        print("[sim] 跳过 {}：未找到最优模型 {}，请先运行 evaluation.py --best-from-dir lido-run-events --exp-name {}".format(
            exp_name, path, exp_name))
        return False

    env = JunctionPetriNetEnv(render_mode=render_mode, reward_function=base_reward,
                              net=get_petri_net('data/traffic-scenario.PNPRO', type=Parser.PNPRO), transitions_to_obs=True, places_to_obs=False)
    env.reset()
    env = wrap_env(env, wrapper="gymnasium")
    
    # 根据模型类型创建对应的 agent（cdqn 或 dqn）
    constrained = (model_type == "cdqn")
    agent = get_dqn_model(env, memory=None, cfg=DQN_DEFAULT_CONFIG.copy(), constrained=constrained)

    agent.load(path)
    agent.set_mode("eval")
    agent.set_running_mode("eval")

    terminated = False
    obs, _ = env.reset()
    #t = 0
    #while not terminated:
    #    action = torch.argmax(agent.q_network.act({"states": obs}, role="q_network")[0], dim=1, keepdim=True)

    #    obs, reward, terminated, truncated, info = env.step(action)
    #    print(reward)
    #    env.render()

    total_timesteps = []
    constraints_broken = []
    average_waiting_times = []
    min_waiting_times = []
    max_waiting_times = []
    for i in range(iterations):
        obs, _ = env.reset()
        c = 0
        all_waiting_times = []  # 记录所有车的等待时间（包括通过的和未通过的）
        t = 0
        terminated = False
        while not terminated:
            # 修改：使用 agent.act() 而不是手动 argmax
            # 这能确保使用正确的预处理和 CDQN 的内部逻辑
            action, _, _ = agent.act(obs, timestep=0, timesteps=0)

            obs, reward, terminated, _, inf = env.step(action)
            if not inf["success"]:
                c = c + 1
            # 收集本步通过的车的等待时间
            all_waiting_times.extend(inf["waiting_times"])
            t = t + 1
        
        # 游戏结束，统计还在队列里的车（未通过）
        for lane in env.unwrapped.lanes:
            for vehicle in lane.vehicles:
                all_waiting_times.append(vehicle.time_steps)
        
        # 调试：输出前几轮的统计
        if i < 3:
            print(f"  [轮 {i+1}] 步数={t}, 违规={c}, 总车辆数={len(all_waiting_times)}")
        
        total_timesteps.append(t)
        constraints_broken.append(c)
        
        # 统计所有车的等待时间
        if len(all_waiting_times) > 0:
            average_waiting_times.append(sum(all_waiting_times) / len(all_waiting_times))
            min_waiting_times.append(min(all_waiting_times))
            max_waiting_times.append(max(all_waiting_times))
        else:
            # 真的一辆车都没有（极端情况）
            average_waiting_times.append(0)
            min_waiting_times.append(0)
            max_waiting_times.append(0)

    avg_timesteps = float(sum(total_timesteps)) / float(iterations)
    min_timesteps = min(total_timesteps)
    max_timesteps = max(total_timesteps)
    max_steps = max(total_timesteps) if total_timesteps else 1
    min_constraints_broken = min(constraints_broken) / float(max_steps) * 100 if constraints_broken else 0
    avg_constraints_broken = sum(constraints_broken) / len(constraints_broken) / float(max_steps) * 100 if constraints_broken else 0
    max_constraints_broken = max(constraints_broken) / float(max_steps) * 100 if constraints_broken else 0
    min_waiting_time = float(sum(min_waiting_times)) / float(len(min_waiting_times)) if min_waiting_times else 0
    avg_waiting_time = float(sum(average_waiting_times)) / float(len(average_waiting_times)) if average_waiting_times else 0
    max_waiting_time = float(sum(max_waiting_times)) / float(len(max_waiting_times)) if max_waiting_times else 0
    # 仅输出该组参数的汇总（不打印整列原始数据）
    print("[sim] type={} | path={} | avg_steps={:.0f} | avg_c_broken={:.2f}% | avg_wait={:.2f}".format(
        model_type.upper(), path, avg_timesteps, avg_constraints_broken, avg_waiting_time))

    df.loc[len(df.index)] = [model_type, p[0], p[1], p[2], p[3], p[4], min_timesteps, avg_timesteps, max_timesteps,
                             min_waiting_time, avg_waiting_time, max_waiting_time, min_constraints_broken,
                             avg_constraints_broken, max_constraints_broken]
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="RLPN-Simulation")
    parser.add_argument("--model-type", type=str, default="both",
                        choices=["cdqn", "dqn", "both"],
                        help="评估哪种模型：cdqn（CDQN）、dqn（DQN）、both（两者都评估）")
    parser.add_argument("--iterations", type=int, default=200,
                        help="每组模型评估的 episode 数（默认 200）")
    parser.add_argument("--render", action="store_true", help="开启渲染（默认不渲染以加速批量评估）")
    args = parser.parse_args()
    render_mode = "human" if args.render else None

    if args.model_type == "both":
        for model_type in ["cdqn", "dqn"]:
            print(f"\n{'='*60}\n正在评估 {model_type.upper()} 最优模型 (*_best.pt)\n{'='*60}")
            for p in params:
                main(p, model_type=model_type, render_mode=render_mode, iterations=args.iterations)
    else:
        print(f"\n{'='*60}\n正在评估 {args.model_type.upper()} 最优模型 (*_best.pt)\n{'='*60}")
        for p in params:
            main(p, model_type=args.model_type, render_mode=render_mode, iterations=args.iterations)

    df.to_csv("simulation-results.csv", index=False)
    print(f"\n结果已保存到 simulation-results.csv")
