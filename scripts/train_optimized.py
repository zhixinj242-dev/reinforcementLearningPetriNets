"""
【文件角色】：重构后的训练脚本入口。
负责：解析命令行参数、构建环境与记忆池、配置 DQN/CDQN、启动训练或评估。
特点：优化的违规日志记录（只在环境不接受动作时记录）和奖励曲线生成。
"""
import argparse
import sys
import os
import time
import traceback
from typing import Dict, Any, Optional

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
from utils.optimized_violation_logger import OptimizedViolationLogger
from utils.reward_tracker import RewardTracker
from utils.entities import PetriPlace, PetriTransition, PetriArc


class TrainingConfig:
    """训练配置类，封装所有训练参数"""
    
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.exploration_timesteps = args.exploration_timesteps
        self.exploration_final_epsilon = args.exploration_final_epsilon
        self.learning_starts = args.learning_starts
        self.random_timesteps = args.random_timesteps
        self.timesteps = args.timesteps
        self.constrained = args.constrained
        self.exp_name = args.exp_name or self._generate_experiment_name(args)
        
        # 奖励函数参数
        self.reward_params = {
            'success': args.m_success,
            'cars_driven': args.m_cars_driven,
            'waiting_time': args.m_waiting_time,
            'max_waiting_time': args.m_max_waiting_time,
            'timestep': args.m_timestep,
        }
        
        # 算法名称
        self.algorithm_name = "CDQN" if self.constrained else "DQN"
        
        # 路径配置
        self.data_path = os.path.join(PROJECT_ROOT, "data", "traffic-scenario.PNPRO")
        self.checkpoints_dir = os.path.join(PROJECT_ROOT, "experiments", "checkpoints")
        self.logs_dir = os.path.join(PROJECT_ROOT, "experiments", "logs")
        self.results_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
        
        # 创建目录
        for directory in [self.checkpoints_dir, self.logs_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def _generate_experiment_name(self, args):
        """生成实验名称"""
        constrained_str = "cdqn" if args.constrained else "dqn"
        return f"agent_s{args.m_success}c{args.m_cars_driven}w{args.m_waiting_time}mw{args.m_max_waiting_time}t{args.m_timestep}_{constrained_str}"


def get_petri_net(file_path: str, net_type: str):
    """获取Petri网的函数，替代 utils.petri_net"""
    if net_type == "PNPRO":
        parser = PNProParser(file_path)
        # 调用正确的解析方法
        parser._get_petri_net_entities()
        return parser._generate_snake_petri_net()
    else:
        raise ValueError(f"不支持的Petri网类型: {net_type}")


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


class OptimizedTrainer:
    """优化的训练器，集成违规日志和奖励跟踪"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.env = None
        self.agent = None
        self.trainer = None
        self.violation_logger = None
        self.reward_tracker = None
        
        # 训练状态
        self.current_episode = 0
        self.current_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
    def setup_environment(self):
        """设置环境"""
        try:
            # 获取奖励函数
            reward_function = get_reward_function argparse.Namespace(
                constrained=self.config.constrained,
                m_success=self.config.reward_params['success'],
                m_cars_driven=self.config.reward_params['cars_driven'],
                m_waiting_time=self.config.reward_params['waiting_time'],
                m_max_waiting_time=self.config.reward_params['max_waiting_time'],
                m_timestep=self.config.reward_params['timestep']
            )
            
            # 创建环境
            self.env = JunctionPetriNetEnv(
                reward_function=reward_function,
                net=get_petri_net(self.config.data_path, type="PNPRO"),
                transitions_to_obs=True,
                places_to_obs=False,
            )
            self.env.reset()
            self.env = wrap_env(self.env, wrapper="gymnasium")
            
            print(f"✓ 环境设置成功")
            return True
            
        except Exception as e:
            print(f"✗ 环境设置失败: {e}")
            traceback.print_exc()
            return False
    
    def setup_agent(self):
        """设置智能体"""
        try:
            # 构建智能体
            memory = RandomMemory(memory_size=500000, num_envs=self.env.num_envs)
            cfg = DQN_DEFAULT_CONFIG.copy()
            cfg["experiment"]["directory"] = self.config.checkpoints_dir
            cfg["experiment"]["experiment_name"] = self.config.exp_name
            cfg["experiment"]["write_interval"] = 500
            
            # 创建日志记录器和奖励跟踪器
            self.violation_logger = OptimizedViolationLogger(
                self.config.algorithm_name, 
                self.config.reward_params, 
                self.config.logs_dir
            )
            
            self.reward_tracker = RewardTracker(
                self.config.algorithm_name,
                self.config.reward_params,
                self.config.results_dir
            )
            
            # 创建智能体
            self.agent = get_dqn_model(
                env=self.env, 
                memory=memory, 
                cfg=cfg, 
                constrained=self.config.constrained,
                log_manager=self.violation_logger,
                raw_env=self.env.unwrapped
            )
            
            print(f"✓ 智能体设置成功")
            return True
            
        except Exception as e:
            print(f"✗ 智能体设置失败: {e}")
            traceback.print_exc()
            return False
    
    def setup_trainer(self):
        """设置训练器"""
        try:
            # 创建训练器
            self.trainer = SequentialTrainer(
                cfg=self.agent.cfg,
                env=self.env,
                agents=self.agent
            )
            
            print(f"✓ 训练器设置成功")
            return True
            
        except Exception as e:
            print(f"✗ 训练器设置失败: {e}")
            traceback.print_exc()
            return False
    
    def train(self):
        """执行训练"""
        try:
            print(f"\n=== 开始训练 {self.config.algorithm_name} ===")
            print(f"实验名称: {self.config.exp_name}")
            print(f"奖励参数: {self.config.reward_params}")
            print(f"训练步数: {self.config.timesteps}")
            print(f"项目根目录: {PROJECT_ROOT}")
            print(f"源代码目录: {SRC_DIR}")
            print(f"数据文件: {self.config.data_path}")
            print(f"检查点目录: {self.config.checkpoints_dir}")
            print(f"日志目录: {self.config.logs_dir}")
            print(f"结果目录: {self.config.results_dir}")
            print("=" * 50)
            
            # 记录开始时间
            start_time = time.time()
            
            # 执行训练
            self.trainer.train(self.config.timesteps)
            
            # 计算训练时间
            training_time = time.time() - start_time
            
            print(f"\n=== {self.config.algorithm_name} 训练完成 ===")
            print(f"训练时间: {training_time:.2f} 秒")
            print(f"平均每步时间: {training_time/self.config.timesteps:.4f} 秒")
            
            return True
            
        except Exception as e:
            print(f"\n✗ 训练过程中出现错误: {e}")
            traceback.print_exc()
            return False
        
        finally:
            # 确保关闭日志记录器和奖励跟踪器
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.violation_logger:
                self.violation_logger.close()
            if self.reward_tracker:
                self.reward_tracker.close()
            print("✓ 资源清理完成")
        except Exception as e:
            print(f"✗ 资源清理失败: {e}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(prog="RLPN-Train-Optimized")
    
    # --- 运行模式参数 ---
    parser.add_argument("-t", "--train", action='store_true', default=False,
                       help="训练模式")
    parser.add_argument("-e", "--eval", action='store_true', default=False,
                       help="评估模式")
    parser.add_argument("-p", "--path", type=str, default=None,
                       help="模型路径（用于评估）")
    
    # --- 【超参数】---
    parser.add_argument("-b", "--batch-size", type=int, default=64,
                       help="批次大小")
    parser.add_argument("-exp-t", "--exploration-timesteps", type=int, default=2000,
                       help="探索时间步数")
    parser.add_argument("-exp-e", "--exploration-final-epsilon", type=float, default=0.04,
                       help="最终探索率")
    parser.add_argument("-learn-s", "--learning-starts", type=int, default=500,
                       help="开始学习的步数")
    parser.add_argument("-rand-t", "--random-timesteps", type=int, default=500,
                       help="随机探索步数")

    # --- 奖励函数参数 ---
    parser.add_argument("--reward-function", type=str, default="dynamic_reward",
                       help="奖励函数类型")
    parser.add_argument("--m-success", type=float, default=1.0,
                       help="成功奖励权重")
    parser.add_argument("--m-cars-driven", type=float, default=1.0,
                       help="通行量奖励权重")
    parser.add_argument("--m-waiting-time", type=float, default=1.0,
                       help="等待时间惩罚权重")
    parser.add_argument("--m-max-waiting-time", type=float, default=1.0,
                       help="最长等待时间惩罚权重")
    parser.add_argument("--m-timestep", type=float, default=1.0,
                       help="时间步奖励权重")
    
    # --- CDQN vs DQN ---
    parser.add_argument("--constrained", action='store_true', default=False,
                       help="使用CDQN（约束深度Q网络）")
    parser.add_argument("--no-constrained", dest='constrained', action='store_false',
                       help="使用DQN（标准深度Q网络）")
    
    # --- 其他参数 ---
    parser.add_argument("--timesteps", type=int, default=3000,
                       help="训练步数")
    parser.add_argument("--exp-name", type=str, default=None,
                       help="实验名称")
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 创建配置
    config = TrainingConfig(args)
    
    # 创建训练器
    trainer = OptimizedTrainer(config)
    
    # 设置环境
    if not trainer.setup_environment():
        return 1
    
    # 设置智能体
    if not trainer.setup_agent():
        return 1
    
    # 设置训练器
    if not trainer.setup_trainer():
        return 1
    
    # 执行训练
    if args.train:
        if not trainer.train():
            return 1
    
    # 评估模式（待实现）
    elif args.eval and args.path:
        print("评估模式尚未实现")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())