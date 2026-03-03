"""
【文件角色】：最终优化版本的训练脚本入口。
集成所有优化组件：违规日志记录、奖励曲线生成、错误处理和调试机制。
特点：只在环境不接受动作时记录违规，提供详细的奖励曲线和全面的错误处理。
"""
import argparse
import sys
import os
import time
import traceback
from typing import Dict, Any, Optional, Tuple

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
from utils.debug_logger import DebugLogger, ErrorRecovery, ErrorCategory, ErrorSeverity
from utils.entities import PetriPlace, PetriTransition, PetriArc


class FinalOptimizedTrainer:
    """最终优化的训练器，集成所有优化组件"""
    
    def __init__(self):
        # 初始化调试日志记录器
        self.debug_logger = DebugLogger()
        self.error_recovery = ErrorRecovery(self.debug_logger)
        
        # 训练组件
        self.config = None
        self.env = None
        self.agent = None
        self.trainer = None
        self.violation_logger = None
        self.reward_tracker = None
        
        # 训练状态
        self.current_episode = 0
        self.current_step = 0
        self.training_start_time = None
        
        self.debug_logger.log_info("最终优化训练器已初始化")
    
    def setup_environment(self, config) -> bool:
        """设置环境"""
        try:
            self.debug_logger.log_info("开始设置环境", {"config": config.__dict__})
            
            # 获取奖励函数
            reward_function = get_reward_function(config)
            
            # 创建环境
            self.env = JunctionPetriNetEnv(
                reward_function=reward_function,
                net=get_petri_net(config.data_path, type="PNPRO"),
                transitions_to_obs=True,
                places_to_obs=False,
            )
            self.env.reset()
            self.env = wrap_env(self.env, wrapper="gymnasium")
            
            self.debug_logger.log_info("环境设置成功", {
                "action_space": str(self.env.action_space),
                "observation_space": str(self.env.observation_space)
            })
            return True
            
        except Exception as e:
            self.debug_logger.log_error(
                e, 
                ErrorCategory.ENVIRONMENT, 
                ErrorSeverity.HIGH,
                {"config": config.__dict__ if config else {}}
            )
            
            # 尝试恢复
            recovered, message = self.error_recovery.handle_error(e, ErrorCategory.ENVIRONMENT)
            self.debug_logger.log_info(f"环境错误恢复尝试: {recovered}, {message}")
            
            return False
    
    def setup_agent(self, config) -> bool:
        """设置智能体"""
        try:
            self.debug_logger.log_info("开始设置智能体", {"config": config.__dict__})
            
            # 构建智能体
            memory = RandomMemory(memory_size=500000, num_envs=self.env.num_envs)
            cfg = DQN_DEFAULT_CONFIG.copy()
            cfg["experiment"]["directory"] = config.checkpoints_dir
            cfg["experiment"]["experiment_name"] = config.exp_name
            cfg["experiment"]["write_interval"] = 500
            
            # 创建日志记录器和奖励跟踪器
            self.violation_logger = OptimizedViolationLogger(
                config.algorithm_name, 
                config.reward_params, 
                config.logs_dir
            )
            
            self.reward_tracker = RewardTracker(
                config.algorithm_name,
                config.reward_params,
                config.results_dir
            )
            
            # 创建智能体
            self.agent = get_dqn_model(
                env=self.env, 
                memory=memory, 
                cfg=cfg, 
                constrained=config.constrained,
                log_manager=self.violation_logger,
                raw_env=self.env.unwrapped
            )
            
            self.debug_logger.log_info("智能体设置成功", {
                "algorithm": config.algorithm_name,
                "constrained": config.constrained,
                "memory_size": memory.memory_size
            })
            return True
            
        except Exception as e:
            self.debug_logger.log_error(
                e, 
                ErrorCategory.AGENT, 
                ErrorSeverity.HIGH,
                {"config": config.__dict__ if config else {}}
            )
            
            # 尝试恢复
            recovered, message = self.error_recovery.handle_error(e, ErrorCategory.AGENT)
            self.debug_logger.log_info(f"智能体错误恢复尝试: {recovered}, {message}")
            
            return False
    
    def setup_trainer(self, config) -> bool:
        """设置训练器"""
        try:
            self.debug_logger.log_info("开始设置训练器", {"config": config.__dict__})
            
            # 创建训练器
            self.trainer = SequentialTrainer(
                cfg=self.agent.cfg,
                env=self.env,
                agents=self.agent
            )
            
            self.debug_logger.log_info("训练器设置成功", {
                "trainer_type": type(self.trainer).__name__
            })
            return True
            
        except Exception as e:
            self.debug_logger.log_error(
                e, 
                ErrorCategory.TRAINING, 
                ErrorSeverity.HIGH,
                {"config": config.__dict__ if config else {}}
            )
            
            # 尝试恢复
            recovered, message = self.error_recovery.handle_error(e, ErrorCategory.TRAINING)
            self.debug_logger.log_info(f"训练器错误恢复尝试: {recovered}, {message}")
            
            return False
    
    def train(self, config) -> bool:
        """执行训练"""
        try:
            self.config = config
            self.training_start_time = time.time()
            
            self.debug_logger.log_info("开始训练", {
                "algorithm": config.algorithm_name,
                "experiment_name": config.exp_name,
                "reward_params": config.reward_params,
                "timesteps": config.timesteps
            })
            
            print(f"\n=== 开始训练 {config.algorithm_name} ===")
            print(f"实验名称: {config.exp_name}")
            print(f"奖励参数: {config.reward_params}")
            print(f"训练步数: {config.timesteps}")
            print(f"项目根目录: {PROJECT_ROOT}")
            print(f"源代码目录: {SRC_DIR}")
            print(f"数据文件: {config.data_path}")
            print(f"检查点目录: {config.checkpoints_dir}")
            print(f"日志目录: {config.logs_dir}")
            print(f"结果目录: {config.results_dir}")
            print("=" * 50)
            
            # 执行训练
            self.trainer.train(config.timesteps)
            
            # 计算训练时间
            training_time = time.time() - self.training_start_time
            
            print(f"\n=== {config.algorithm_name} 训练完成 ===")
            print(f"训练时间: {training_time:.2f} 秒")
            print(f"平均每步时间: {training_time/config.timesteps:.4f} 秒")
            
            # 获取奖励统计
            reward_stats = self.reward_tracker.get_statistics()
            print(f"\n=== 奖励统计 ===")
            for key, value in reward_stats.items():
                print(f"{key}: {value}")
            
            self.debug_logger.log_info("训练完成", {
                "training_time": training_time,
                "avg_step_time": training_time/config.timesteps,
                "reward_stats": reward_stats
            })
            
            return True
            
        except Exception as e:
            self.debug_logger.log_error(
                e, 
                ErrorCategory.TRAINING, 
                ErrorSeverity.CRITICAL,
                {"config": config.__dict__ if config else {}}
            )
            
            # 尝试恢复
            recovered, message = self.error_recovery.handle_error(e, ErrorCategory.TRAINING)
            self.debug_logger.log_info(f"训练错误恢复尝试: {recovered}, {message}")
            
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
            self.debug_logger.log_info("资源清理完成")
        except Exception as e:
            self.debug_logger.log_error(
                e, 
                ErrorCategory.SYSTEM, 
                ErrorSeverity.MEDIUM,
                {"operation": "cleanup"}
            )


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


def get_reward_function(config):
    """根据参数获取奖励函数"""
    if config.constrained:
        # CDQN 使用带约束的奖励函数
        return lambda prev_obs, obs, success, timestep, waiting_times: (
            rewards.constraint_driven_waiting_times_timesteps(prev_obs, obs, success, timestep, waiting_times) * config.reward_params['success'] +
            rewards.constraint_cars_driven_timestep(prev_obs, obs, success, timestep, waiting_times) * config.reward_params['cars_driven'] +
            rewards.constraint_avg_waiting_times_and_timesteps(prev_obs, obs, success, timestep, waiting_times) * config.reward_params['waiting_time'] +
            rewards.constraint_timestep(prev_obs, obs, success, timestep, waiting_times) * config.reward_params['max_waiting_time'] +
            rewards.constraint_timestep(prev_obs, obs, success, timestep, waiting_times) * config.reward_params['timestep']
        )
    else:
        # DQN 使用不带约束的奖励函数
        return lambda prev_obs, obs, success, timestep, waiting_times: (
            rewards.driven_waiting_times_timesteps(prev_obs, obs, success, timestep, waiting_times) * config.reward_params['success'] +
            rewards.cars_driven_timestep(prev_obs, obs, success, timestep, waiting_times) * config.reward_params['cars_driven'] +
            rewards.avg_waiting_times_and_timesteps(prev_obs, obs, success, timestep, waiting_times) * config.reward_params['waiting_time'] +
            rewards.timestep(prev_obs, obs, success, timestep, waiting_times) * config.reward_params['max_waiting_time'] +
            rewards.timestep(prev_obs, obs, success, timestep, waiting_times) * config.reward_params['timestep']
        )


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(prog="RLPN-Train-Final-Optimized")
    
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
    # 创建训练器
    trainer = FinalOptimizedTrainer()
    
    try:
        # 解析参数
        args = parse_arguments()
        trainer.debug_logger.log_info("命令行参数解析完成", {"args": vars(args)})
        
        # 创建配置
        config = TrainingConfig(args)
        trainer.debug_logger.log_info("训练配置创建完成", {"config": config.__dict__})
        
        # 检查环境状态
        env_status = trainer.debug_logger.check_environment()
        trainer.debug_logger.log_info("环境检查完成", {"env_status": env_status})
        
        # 设置环境
        if not trainer.setup_environment(config):
            trainer.debug_logger.log_error(
                Exception("环境设置失败"), 
                ErrorCategory.ENVIRONMENT, 
                ErrorSeverity.CRITICAL
            )
            return 1
        
        # 设置智能体
        if not trainer.setup_agent(config):
            trainer.debug_logger.log_error(
                Exception("智能体设置失败"), 
                ErrorCategory.AGENT, 
                ErrorSeverity.CRITICAL
            )
            return 1
        
        # 设置训练器
        if not trainer.setup_trainer(config):
            trainer.debug_logger.log_error(
                Exception("训练器设置失败"), 
                ErrorCategory.TRAINING, 
                ErrorSeverity.CRITICAL
            )
            return 1
        
        # 执行训练
        if args.train:
            if not trainer.train(config):
                trainer.debug_logger.log_error(
                    Exception("训练失败"), 
                    ErrorCategory.TRAINING, 
                    ErrorSeverity.CRITICAL
                )
                return 1
        
        # 评估模式（待实现）
        elif args.eval and args.path:
            trainer.debug_logger.log_info("评估模式尚未实现")
            print("评估模式尚未实现")
            return 1
        
        return 0
        
    except Exception as e:
        trainer.debug_logger.log_error(
            e, 
            ErrorCategory.SYSTEM, 
            ErrorSeverity.CRITICAL,
            {"phase": "main_execution"}
        )
        return 1
    
    finally:
        # 确保关闭调试日志记录器
        trainer.debug_logger.close()


if __name__ == "__main__":
    sys.exit(main())