"""
【文件角色】：已训练模型评估脚本。
1) 单模型评估：-p path 加载指定 .pt 跑若干步。
2) 选最优 checkpoint：--best-from-dir + --exp-name 从带步数的 checkpoint 中评估选最优，保存为 *_best.pt。
"""
import argparse
import glob
import os
import shutil

import torch
from skrl.agents.torch.dqn import DQN_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer

from agents.dqn import get_dqn_model
from environment import JunctionPetriNetEnv
import rewards
from utils.petri_net import get_petri_net, Parser


def generate_parsed_arguments():
    """解析评估所需的参数"""
    parser = argparse.ArgumentParser(prog="RLPN-Eval")
    parser.add_argument("-p", "--path", type=str, default=None, help="单模型评估：要评估的 .pt 路径")
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("--best-from-dir", type=str, default=None, help="选最优：实验目录，如 lido-run-events")
    parser.add_argument("--exp-name", type=str, default=None, help="选最优：实验名，如 agent_s1.0c0w1mw0t0_cdqn")
    parser.add_argument("--constrained", action="store_true", default=True)
    parser.add_argument("--no-constrained", dest="constrained", action="store_false")
    parser.add_argument("--best-eval-steps", type=int, default=3000, help="选最优时每个 checkpoint 评估步数")
    parser.add_argument("--best-eval-episodes", type=int, default=10, help="选最优时每个 checkpoint 评估 episode 数（用于算平均）")
    parser.add_argument("--max-steps-per-episode", type=int, default=5000, help="单 episode 最大步数，防止环境不终止导致卡死")
    return parser.parse_args()


def _make_eval_env():
    env = JunctionPetriNetEnv(
        reward_function=rewards.base_reward,
        net=get_petri_net('data/traffic-scenario.PNPRO', type=Parser.PNPRO),
    )
    env.reset()
    return wrap_env(env, wrapper="gymnasium")


def _eval_one_checkpoint(ckpt_path, constrained, best_eval_episodes, max_steps_per_episode=5000):
    """对单个 checkpoint 跑若干 episode，返回平均生存步数（越大越好）。"""
    env = _make_eval_env()
    memory = RandomMemory(memory_size=50000, num_envs=env.num_envs)
    cfg = DQN_DEFAULT_CONFIG.copy()
    cfg["batch_size"] = 64
    cfg["exploration"]["timesteps"] = 0
    cfg["exploration"]["final_epsilon"] = 0.0
    cfg["random_timesteps"] = 0
    cfg["experiment"]["experiment_name"] = "eval_tmp"
    agent = get_dqn_model(env=env, memory=memory, cfg=cfg, constrained=constrained)
    agent.load(ckpt_path)
    agent.set_mode("eval")
    agent.set_running_mode("eval")

    steps_list = []
    for ep in range(best_eval_episodes):
        obs, _ = env.reset()
        t = 0
        terminated = False
        while not terminated and t < max_steps_per_episode:
            action = torch.argmax(agent.q_network.act({"states": obs}, role="q_network")[0], dim=1, keepdim=True)
            obs, _, terminated, _, _ = env.step(action)
            t += 1
        steps_list.append(t)
    return sum(steps_list) / len(steps_list) if steps_list else 0.0


def run_best_from_checkpoints(parser):
    """从带步数的 checkpoint 中选出最优，保存为 exp_name_best.pt。"""
    run_dir = parser.best_from_dir
    exp_name = parser.exp_name
    if not run_dir or not exp_name:
        raise ValueError("选最优模式需同时指定 --best-from-dir 和 --exp-name")

    exp_dir = os.path.join(run_dir, exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    checkpoints = []
    if os.path.isdir(ckpt_dir):
        checkpoints = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    if not checkpoints:
        checkpoints = sorted(glob.glob(os.path.join(exp_dir, "agent_*.pt")))
    if not checkpoints:
        checkpoints = sorted(glob.glob(os.path.join(run_dir, "agent_*.pt")))
    if not checkpoints:
        checkpoints = sorted(glob.glob(os.path.join(run_dir, "agent_*_agent_final.pt")))
    if not checkpoints:
        print(f"[evaluation] 未找到带步数 checkpoint：{exp_dir} 或 {ckpt_dir} 或 {run_dir}")
        return

    best_path = None
    best_avg_steps = -1.0
    n_ckpts = len(checkpoints)
    for i, ckpt in enumerate(checkpoints):
        name = os.path.basename(ckpt)
        print(f"[evaluation] 正在评估 checkpoint {i+1}/{n_ckpts}: {name} ...", flush=True)
        avg = _eval_one_checkpoint(ckpt, parser.constrained, parser.best_eval_episodes, parser.max_steps_per_episode)
        print(f"  {name} -> 平均步数 {avg:.0f}", flush=True)
        if avg > best_avg_steps:
            best_avg_steps = avg
            best_path = ckpt

    if best_path is None:
        print(f"[evaluation] 未找到任何 checkpoint，使用最终模型", flush=True)
        # 尝试查找最终模型
        final_model_path = os.path.join(run_dir, f"{exp_name}/checkpoints/agent_{exp_name}_agent_final.pt")
        if os.path.isfile(final_model_path):
            best_path = final_model_path
            best_avg_steps = 0
        else:
            print(f"[evaluation] 也未找到最终模型，无法生成 best.pt", flush=True)
            return
    out_path = os.path.join(run_dir, f"{exp_name}_best.pt")
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy2(best_path, out_path)
    print(f"[evaluation] 最优 checkpoint 已保存为: {out_path} (平均步数 {best_avg_steps:.0f})")


def main():
    parser = generate_parsed_arguments()

    if parser.best_from_dir is not None and parser.exp_name is not None:
        run_best_from_checkpoints(parser)
        return

    if parser.path is None:
        print("请指定 -p/--path 进行单模型评估，或 --best-from-dir + --exp-name 选最优 checkpoint")
        return

    env = _make_eval_env()
    memory = RandomMemory(memory_size=500000, num_envs=env.num_envs)
    cfg = DQN_DEFAULT_CONFIG.copy()
    cfg["batch_size"] = parser.batch_size
    cfg["exploration"]["timesteps"] = 0
    cfg["exploration"]["final_epsilon"] = 0.0
    cfg["random_timesteps"] = 0
    cfg["experiment"]["experiment_name"] = "eval_{}".format(
        parser.path.replace("/", "_").replace("\\", "_").replace(".pt", "") if parser.path else "eval_run"
    )
    dqn_agent = get_dqn_model(env=env, memory=memory, cfg=cfg, constrained=parser.constrained)
    dqn_agent.load(parser.path)
    cfg_trainer = {"timesteps": 5000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=dqn_agent)
    trainer.eval()


if __name__ == '__main__':
    main()
