"""
【文件角色】：训练脚本入口。
负责：解析命令行参数、构建环境与记忆池、配置 DQN/CDQN、启动训练或评估。
训练只保存带步数的 checkpoint（由 skrl checkpoint_interval 写入）；
最优模型由 evaluation.py --best-from-dir --exp-name 从 checkpoint 中选出并保存为 *_best.pt，供 simulation.py / visual.py 使用。
"""
import argparse
import itertools

from skrl.agents.torch.dqn import DQN_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer

from agents.dqn import get_dqn_model
from environment import JunctionPetriNetEnv
import rewards
from utils.petri_net import get_petri_net, Parser
from utils.log_manager import LogManager


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
    parser.add_argument("--reward-function", type=str, default="discounted_reward")
    parser.add_argument("--m-success", type=float, default=1.0)
    parser.add_argument("--m-cars-driven", type=float, default=1.0)
    parser.add_argument("--m-waiting-time", type=float, default=1.0)
    parser.add_argument("--m-max-waiting-time", type=float, default=1.0)
    parser.add_argument("--m-timestep", type=float, default=1.0)
    
    # --- CDQN vs DQN ---
    parser.add_argument("--constrained", action='store_true', default=False)
    parser.add_argument("--no-constrained", dest='constrained', action='store_false')

    # --- 多进程标识 ---
    parser.add_argument("--process-id", type=int, default=None)

    return parser.parse_args()


def main():
    """总指挥。负责把环境、AI、记忆池串联起来，并启动训练。"""
    parser = generate_parsed_arguments()

    pid_prefix = f"[P{parser.process_id:02d}]" if parser.process_id is not None else ""

    # 1. 选择奖励函数
    reward_fn_name = parser.reward_function
    reward_fn = getattr(rewards, reward_fn_name, rewards.discounted_reward)
    if reward_fn_name == "dynamic_reward":
        rewards.success_multiplier = parser.m_success
        rewards.car_driven_multiplier = parser.m_cars_driven
        rewards.waiting_time_multiplier = parser.m_waiting_time
        rewards.max_waiting_time_multiplier = parser.m_max_waiting_time
        rewards.timestep_multiplier = parser.m_timestep

    # 2. 创建奖励函数参数字典
    reward_params = {
        "success": parser.m_success,
        "cars_driven": parser.m_cars_driven,
        "waiting_time": parser.m_waiting_time,
        "max_waiting_time": parser.m_max_waiting_time,
        "timestep": parser.m_timestep
    }

    # 3. 创建LogManager实例
    algorithm_type = "CDQN" if parser.constrained else "DQN"
    log_manager = LogManager(algorithm_type=algorithm_type, reward_params=reward_params)
    print(f"{pid_prefix} 日志文件: {log_manager.get_log_file_name()}")

    # 4. 初始化环境
    env = JunctionPetriNetEnv(
        reward_function=reward_fn,
        net=get_petri_net('data/traffic-scenario.PNPRO', type=Parser.PNPRO),
        log_manager=log_manager
    )
    # 添加算法类型标识
    env.algorithm_type = algorithm_type
    env.reset()
    
    # 设置设备为 GPU（如果可用），否则使用 CPU
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    # 确保环境也使用相同的设备
    env.device = device
    
    # 现在包装环境
    env = wrap_env(env, wrapper="gymnasium")
    # 确保包装后的环境也使用相同的设备
    env.device = device

    # 5. 设置记忆池
    memory = RandomMemory(memory_size=500000, num_envs=env.num_envs)
    # 确保记忆池也使用相同的设备
    memory.device = device

    state = "{}{}".format("train" if parser.train else "", "eval" if parser.eval else "")

    # 6. 算法配置
    cfg = DQN_DEFAULT_CONFIG.copy()
    cfg["batch_size"] = parser.batch_size
    cfg["exploration"]["timesteps"] = parser.exploration_timesteps
    cfg["exploration"]["final_epsilon"] = parser.exploration_final_epsilon
    cfg["learning_starts"] = parser.learning_starts
    cfg["random_timesteps"] = parser.random_timesteps
    cfg["learning_rate"] = 1e-4  # 添加学习率设置
    
    cfg["experiment"]["checkpoint_interval"] = 500
    cfg["experiment"]["write_interval"] = 100
    
    # 生成实验名称
    suffix = "_cdqn" if parser.constrained else "_dqn"
    cfg["experiment"]["experiment_name"] = "agent_s{}c{}w{}mw{}t{}{}".format(
        parser.m_success, parser.m_cars_driven, parser.m_waiting_time,
        parser.m_max_waiting_time, parser.m_timestep, suffix
    )

    # 7. 创建 AI 模型
    try:
        # 获取原始环境（在wrap_env之前）
        raw_env = env.unwrapped if hasattr(env, 'unwrapped') else env
        
        # 将设备传递给 get_dqn_model 函数
        agent = get_dqn_model(env=env, memory=memory, cfg=cfg, constrained=parser.constrained, log_manager=log_manager, device=device, raw_env=raw_env)
        # 确保代理也使用 CPU
        agent.device = device
        if parser.path is not None:
            agent.load(parser.path)
    except Exception as e:
        print(f"{pid_prefix} ✗ FAILED to create agent: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        if log_manager:
            log_manager.close()
        return

    # 8. 设置训练器
    # 【关键修复】：确保 lido-run-events 目录存在
    import os
    os.makedirs("lido-run-events", exist_ok=True)
    
    exp_name = cfg["experiment"]["experiment_name"]
    if parser.process_id is not None:
        print(f"{pid_prefix} START: {exp_name[:40]}...", flush=True)
    
    # 【关键修复】：使用 skrl 的 SequentialTrainer，但设置 checkpoint 保存路径
    cfg_trainer = {
        "timesteps": 3000,
        "headless": True
    }
    
    print(f"{pid_prefix} 配置 checkpoint 保存目录: lido-run-events", flush=True)
    
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    
    # 【关键修复】：设置 checkpoint 保存路径
    checkpoint_dir = os.path.join("lido-run-events", exp_name, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 【关键修复】：重写 trainer 的训练方法以保存 checkpoint
    original_train = trainer.train
    
    def custom_train():
        # 调用原始训练方法
        original_train()
        
        # 训练完成后，手动保存最终模型
        final_checkpoint_path = os.path.join(checkpoint_dir, f"agent_{exp_name}_agent_final.pt")
        agent.save(final_checkpoint_path)
        print(f"{pid_prefix} 手动保存最终模型: {final_checkpoint_path}", flush=True)
        
        # 训练完成后，手动复制 checkpoint 文件
        import glob
        import shutil
        
        # 【关键修复】：查找所有可能的 checkpoint 文件位置
        checkpoint_files = []
        
        # 位置1：skrl 默认保存位置
        skrl_checkpoints = glob.glob(os.path.join(exp_name, "checkpoints", "*.pt"))
        checkpoint_files.extend(skrl_checkpoints)
        
        # 位置2：当前目录
        current_dir_checkpoints = glob.glob(os.path.join(exp_name + "_agent_*.pt"))
        checkpoint_files.extend(current_dir_checkpoints)
        
        # 位置3：agent_*.pt 格式
        agent_checkpoints = glob.glob("agent_*.pt")
        checkpoint_files.extend(agent_checkpoints)
        
        print(f"{pid_prefix} 找到 {len(checkpoint_files)} 个 checkpoint 文件", flush=True)
        
        # 复制所有 checkpoint 文件到 lido-run-events
        for ckpt_file in checkpoint_files:
            dest_file = os.path.join(checkpoint_dir, os.path.basename(ckpt_file))
            shutil.copy2(ckpt_file, dest_file)
            print(f"{pid_prefix} 复制 checkpoint: {ckpt_file} -> {dest_file}", flush=True)
    
    # 替换训练方法
    trainer.train = custom_train

    # 9. 开始工作
    if parser.train:
        try:
            trainer.train()
            
            # 训练完成后，手动保存最终模型
            final_checkpoint_path = os.path.join(checkpoint_dir, f"agent_{exp_name}_agent_final.pt")
            agent.save(final_checkpoint_path)
            print(f"{pid_prefix} 手动保存最终模型: {final_checkpoint_path}", flush=True)
            
            # 【关键修复】：训练结束后调用 evaluation.py 生成 best agent
            print(f"{pid_prefix} 训练完成，开始评估最优模型...", flush=True)
            
            # 导入 evaluation 模块
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            
            # 调用 evaluation.py 生成 best agent
            import evaluation
            
            # 临时修改 sys.argv 来调用 evaluation
            old_argv = sys.argv
            sys.argv = [
                "evaluation.py",
                "--best-from-dir", "lido-run-events",
                "--exp-name", exp_name,
                "--constrained" if parser.constrained else "--no-constrained"
            ]
            
            try:
                # 生成新的 parser 对象
                eval_parser = evaluation.generate_parsed_arguments()
                evaluation.run_best_from_checkpoints(eval_parser)
                print(f"{pid_prefix} 评估完成，最优模型已保存到: lido-run-events/{exp_name}_best.pt", flush=True)
            except Exception as e:
                print(f"{pid_prefix} 评估失败: {str(e)}", flush=True)
            finally:
                sys.argv = old_argv
            
            # 只保留带步数的 checkpoint（由 skrl checkpoint_interval 保存），不再保存末尾的 .pt
            if parser.process_id is not None:
                print(f"{pid_prefix} ✓ DONE", flush=True)
            print(f"{pid_prefix} 训练完成，日志已保存到: {log_manager.get_log_file_name()}")
        except KeyboardInterrupt:
            if parser.process_id is not None:
                print(f"{pid_prefix} ⊗ STOPPED", flush=True)
        except Exception as e:
            print(f"{pid_prefix} ✗ FAILED: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            # 确保关闭日志文件
            if log_manager:
                log_manager.close()
    if parser.eval:
        trainer.eval()

    print(f"{pid_prefix} ===== 运行完成 =====")


if __name__ == '__main__':
    main()
