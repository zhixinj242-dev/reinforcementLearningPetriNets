"""
将 GAIL 或 CDQN 的评估结果追加到统一对比文件 method_comparison.csv。
与 baseline.py 使用相同的指标和轮数，便于在一张表里对比 B1、B2、GAIL、CDQN。

用法：
  python add_to_comparison.py --method GAIL --path gail_policy.pt
  python add_to_comparison.py --method CDQN --path lido-run-events/agent_s1.5c0.0w1.0mw1.5t0.0_cdqn.pt
  python add_to_comparison.py --method DQN  --path lido-run-events/agent_s1.5c0.0w1.0mw1.5t0.0_dqn.pt
"""
import argparse
import torch
from skrl.agents.torch.dqn import DQN_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory

from environment import JunctionPetriNetEnv
from rewards import base_reward
from utils.petri_net import get_petri_net, Parser
from utils.result_comparison import append_method


def run_episodes_and_compute_metrics(env, get_action_fn, iterations=100):
    """与 baseline.run_baseline 相同的统计逻辑。"""
    total_timesteps = []
    constraints_broken = []
    average_waiting_times = []
    min_waiting_times = []
    max_waiting_times = []

    for i in range(iterations):
        obs, _ = env.reset()
        c = 0
        num_cars_driven = 0
        sum_waiting_times = 0
        min_waiting_time = 10000
        max_waiting_time = 0
        t = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = get_action_fn(obs, t)
            obs, reward, terminated, truncated, inf = env.step(action)

            if not inf["success"]:
                c = c + 1
            num_cars_driven = num_cars_driven + inf["num_cars_driven"]
            sum_waiting_times = sum_waiting_times + sum(inf["waiting_times"])
            if len(inf["waiting_times"]) > 0:
                min_waiting_time = min(min_waiting_time, min(inf["waiting_times"]))
                max_waiting_time = max(max_waiting_time, max(inf["waiting_times"]))
            t = t + 1

        total_timesteps.append(t)
        constraints_broken.append(c)
        if num_cars_driven > 0:
            average_waiting_times.append(float(sum_waiting_times) / float(num_cars_driven))
        if min_waiting_time != 10000:
            min_waiting_times.append(min_waiting_time)
            max_waiting_times.append(max_waiting_time)

    avg_timesteps = sum(total_timesteps) / iterations
    avg_constraints_broken = sum(constraints_broken) / len(constraints_broken) / max(total_timesteps) * 100 if total_timesteps else 0
    avg_waiting_time = sum(average_waiting_times) / len(average_waiting_times) if average_waiting_times else 0
    min_wt = sum(min_waiting_times) / len(min_waiting_times) if min_waiting_times else 0
    max_wt = sum(max_waiting_times) / len(max_waiting_times) if max_waiting_times else 0

    return {
        "avg_timesteps": avg_timesteps,
        "avg_constraints_broken": avg_constraints_broken,
        "avg_waiting_time": avg_waiting_time,
        "min_waiting_time": min_wt,
        "max_waiting_time": max_wt,
    }


def main():
    parser = argparse.ArgumentParser(prog="add_to_comparison")
    parser.add_argument("--method", type=str, required=True, help="方法名，如 GAIL、CDQN、DQN")
    parser.add_argument("--path", type=str, required=True, help="模型文件路径")
    parser.add_argument("--iterations", type=int, default=100, help="评估轮数（与 baseline 一致）")
    args = parser.parse_args()

    env = JunctionPetriNetEnv(
        reward_function=base_reward,
        net=get_petri_net("data/traffic-scenario.PNPRO", type=Parser.PNPRO),
        transitions_to_obs=True,
        places_to_obs=False,
    )
    env.reset()
    env = wrap_env(env, wrapper="gymnasium")

    if args.method.upper() == "GAIL":
        from visual import flatten_obs_to_1d
        ckpt = torch.load(args.path, map_location="cpu")
        obs_dim = int(ckpt["obs_dim"])
        act_dim = int(ckpt["act_dim"])

        class PolicyMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(obs_dim, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, act_dim),
                )
            def forward(self, x):
                return self.net(x)

        policy = PolicyMLP()
        policy.load_state_dict(ckpt["state_dict"])
        policy.eval()

        def get_action(obs, t):
            x = flatten_obs_to_1d(env, obs)
            x_t = torch.from_numpy(x).unsqueeze(0)
            with torch.no_grad():
                logits = policy(x_t)
            return int(torch.argmax(logits, dim=-1).item())

    else:
        # CDQN 或 DQN
        constrained = "CDQN" in args.method.upper()
        memory = RandomMemory(memory_size=500000, num_envs=env.num_envs)
        cfg = DQN_DEFAULT_CONFIG.copy()
        cfg["exploration"]["timesteps"] = 0
        cfg["exploration"]["final_epsilon"] = 0.0
        cfg["random_timesteps"] = 0
        agent = get_dqn_model(env=env, memory=memory, cfg=cfg, constrained=constrained)
        agent.load(args.path)
        agent.set_running_mode("eval")

        from visual import flatten_obs_to_1d
        def get_action(obs, t):
            states_1d = flatten_obs_to_1d(env, obs)
            states = torch.from_numpy(states_1d).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = agent.act(states, timestep=t, timesteps=1)
            return int(action.item())

    print(f"正在评估 {args.method}（{args.iterations} 轮）...")
    metrics = run_episodes_and_compute_metrics(env, get_action, iterations=args.iterations)
    print(f"  平均生存步数: {metrics['avg_timesteps']:.2f}")
    print(f"  平均违规次数: {metrics['avg_constraints_broken']:.2f}%")
    print(f"  平均车辆等待时间: {metrics['avg_waiting_time']:.2f}")

    append_method(args.method, metrics)
    print(f"已写入 method_comparison.csv，可与 B1/B2 等一起查看。")


if __name__ == "__main__":
    main()
