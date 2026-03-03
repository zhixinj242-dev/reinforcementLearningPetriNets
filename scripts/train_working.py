"""
【文件角色】：训练脚本入口。
负责：解析命令行参数、构建环境与记忆池、配置 DQN/CDQN、启动训练或评估。
训练只保存带步数的 checkpoint（由 skrl checkpoint_interval 写入）；
最优模型由 evaluation.py --best-from-dir --exp-name 从 checkpoint 中选出并保存为 *_best.pt，供 simulation.py / visual.py 使用。
"""
import csv
import argparse
import itertools
import sys
import os

# 获取脚本所在目录的父目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

# 添加 src 目录到 Python 路径
sys.path.insert(0, SRC_DIR)

from skrl.agents.torch.dqn import DQN_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer

# 使用绝对导入
from agents.dqn import get_dqn_model
from environment import JunctionPetriNetEnv
import rewards

# 直接导入需要的模块，避免 utils.petri_net 问题
from utils.parser_pnpro import PNProParser
from utils.violation_logger import ViolationLogger
from utils.entities import PetriPlace, PetriTransition, PetriArc


def get_petri_net(file_path, type):
    """获取Petri网的函数，替代 utils.petri_net"""
    if type == "PNPRO":
        parser = PNProParser(file_path)
        return parser.parse()
    else:
        raise ValueError(f"不支持的Petri网类型: {type}")


def generate_parsed_arguments():
    """
    【函数功能】：参数读取器。负责从命令行获取你输入的指令，比如你想跑多少步、记忆池多大。
    """
    parser = argparse.ArgumentParser(prog="RLPN-Train")
    
    # --- 运行模式参数 ---
    parser.add_argument("-t", "--train", action='store_true', default=False)
    parser.add_argument("-e", "--eval", action='store_true', default=False)
    parser.add_argument("-p", "--path", type=str, default=None)
    
    # --- 【超参数】---
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("-exp-t", "--exploration-timesteps", type=int, default=2000)
    parser.add_argument("-exp-e", "--exploration-final-epsilon", type=float, default=0.04)
    parser.add_argument("-learn-s", "--learning-starts", type=int, default=500)
    parser.add_argument("-rand-t", "--random-timesteps", type=int, default=500)

    # --- 奖励函数参数 ---
    parser.add_argument("--reward-function", type=str, default="dynamic_reward")
    parser.add_argument("--m-success", type=float, default=1.0)
    parser.add_argument("--m-cars-driven", type=float, default=1.0)
    parser.add_argument("--m-waiting-time", type=float, default=1.0)
    parser.add_argument("--m-max-waiting-time", type=float, default=1.0)
    parser.add_argument("--m-timestep", type=float, default=1.0)
    
    # --- CDQN vs DQN ---
    parser.add_argument("--constrained", action='store_true', default=False)
    parser.add_argument("--no-constrained", dest='constrained', action='store_false')
    
    # --- 其他参数 ---
    parser.add_argument("--timesteps", type=int, default=3000, help="训练步数")
    parser.add_argument("--exp-name", type=str, default=None, help="实验名称")
    
    return parser.parse_args()


def build_experiment_name(args):
    """构建实验名称"""
    if args.exp_name:
        return args.exp_name
    
    # 自动生成实验名称
    constrained_str = "cdqn" if args.constrained else "dqn"
    return f"agent_s{args.m_success}c{args.m_cars_driven}w{args.m_waiting_time}mw{args.m_max_waiting_time}t{args.m_timestep}_{constrained_str}"


def get_reward_function(args):
    """根据参数获取奖励函数"""
    if args.constrained:
        # CDQN 使用带约束的奖励函数
        return lambda prev_obs, obs, success, timestep, waiting_times: (
            rewards.constraint_driven_waiting_times_timesteps(prev_obs, obs, success, timestep, waiting_times) * args.m_success +
            rewards.constraint_cars_driven_timestep(prev_obs, obs, success, timestep, waiting_times) * args.m_cars_driven +
            rewards.constraint_avg_waiting_times_and_timesteps(prev_obs, obs, success, timestep, waiting_times) * args.m_waiting_time +
            rewards.constraint_timestep(prev_obs, obs, success, timestep, waiting_times) * args.m_max_waiting_time +
            rewards.constraint_timestep(prev_obs, obs, success, timestep, waiting_times) * args.m_timestep
        )
    else:
        # DQN 使用不带约束的奖励函数
        return lambda prev_obs, obs, success, timestep, waiting_times: (
            rewards.driven_waiting_times_timesteps(prev_obs, obs, success, timestep, waiting_times) * args.m_success +
            rewards.cars_driven_timestep(prev_obs, obs, success, timestep, waiting_times) * args.m_cars_driven +
            rewards.avg_waiting_times_and_timesteps(prev_obs, obs, success, timestep, waiting_times) * args.m_waiting_time +
            rewards.timestep(prev_obs, obs, success, timestep, waiting_times) * args.m_max_waiting_time +
            rewards.timestep(prev_obs, obs, success, timestep, waiting_times) * args.m_timestep
        )


def main():
    args = generate_parsed_arguments()
    
    # 构建实验名称
    exp_name = build_experiment_name(args)
    
    # 构建奖励函数参数
    reward_params = {
        'success': args.m_success,
        'cars_driven': args.m_cars_driven,
        'waiting_time': args.m_waiting_time,
        'max_waiting_time': args.m_max_waiting_time,
        'timestep': args.m_timestep,
    }
    
    # 选择奖励函数
    if args.constrained:
        algorithm_name = "CDQN"
    else:
        algorithm_name = "DQN"
    
    # 获取奖励函数
    reward_function = get_reward_function(args)
    
    # 创建违规日志记录器
    violation_logger = ViolationLogger(algorithm_name, reward_params, log_dir="experiments/logs")
    
    # 创建奖励CSV记录器
    csv_file = open('training_rewards.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Episode', 'Reward'])
    
    # 构建环境
    env = JunctionPetriNetEnv(
        reward_function=reward_function,
        net=get_petri_net(os.path.join(PROJECT_ROOT, "data", "traffic-scenario.PNPRO"), type="PNPRO"),
        transitions_to_obs=True,
        places_to_obs=False,
    )
    env.reset()
    env = wrap_env(env, wrapper="gymnasium")
    
    # 构建智能体
    memory = RandomMemory(memory_size=500000, num_envs=env.num_envs)
    cfg = DQN_DEFAULT_CONFIG.copy()
    cfg["experiment"]["directory"] = os.path.join(PROJECT_ROOT, "experiments", "checkpoints")
    cfg["experiment"]["experiment_name"] = exp_name
    cfg["experiment"]["write_interval"] = 500
    
    agent = get_dqn_model(env=env, memory=memory, cfg=cfg, constrained=args.constrained)
    
    # 创建训练器
    trainer = SequentialTrainer(cfg, env, agent)
    
    print(f"开始训练 {algorithm_name}，参数组合: {reward_params}")
    print(f"实验名称: {exp_name}")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"源代码目录: {SRC_DIR}")
    print(f"使用奖励函数: {args.reward_function}")
    
    # 训练循环
    if args.train:
        try:
            trainer.train(args.timesteps)
            print(f"{algorithm_name} 训练完成")
        except Exception as e:
            print(f"训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 关闭日志记录器
            violation_logger.close()
            csv_file.close()
            print("训练日志已保存")
    
    # 评估模式
    elif args.eval and args.path:
        agent.load(args.path)
        agent.set_running_mode("eval")
        print(f"已加载模型: {args.path}")
        
        # 这里可以添加评估逻辑
        # ...


if __name__ == "__main__":
    main()