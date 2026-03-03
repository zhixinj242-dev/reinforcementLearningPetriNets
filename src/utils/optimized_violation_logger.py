import os
import time
import json
import numpy as np

class OptimizedViolationLogger:
    """
    优化的违规日志记录器，只在环境不接受动作时记录
    """
    
    def __init__(self, algorithm_type, reward_params, log_dir="experiments/logs"):
        """
        初始化违规日志记录器
        
        Args:
            algorithm_type: 算法类型，如 "CDQN" 或 "DQN"
            reward_params: 奖励函数参数，字典格式
            log_dir: 日志文件保存目录
        """
        self.algorithm_type = algorithm_type
        self.reward_params = reward_params
        self.log_dir = log_dir
        
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 生成日志文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        params_str = "_".join([f"{k}{v}" for k, v in sorted(reward_params.items())])
        self.log_file_name = f"{log_dir}/violations_{self.algorithm_type}_{params_str}_{timestamp}.log"
        
        # 打开日志文件
        self.log_file = open(self.log_file_name, "w", encoding="utf-8")
        
        # 统计信息
        self.violation_count = 0
        self.total_steps = 0
        self.episode_count = 0
        self.current_episode_violations = 0
        
        # 奖励跟踪
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        
        # 写入日志文件头
        self.write_header()
    
    def write_header(self):
        """写入日志文件头"""
        header = {
            "algorithm_type": self.algorithm_type,
            "reward_params": self.reward_params,
            "log_file_name": self.log_file_name,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "description": "只在环境不接受动作时记录违规"
        }
        self.log_file.write("# VIOLATION LOG HEADER\n")
        self.log_file.write(json.dumps(header, indent=2, ensure_ascii=False))
        self.log_file.write("\n\n# VIOLATION LOGS\n")
        self.log_file.flush()
    
    def log_step(self, step, info):
        """
        记录步骤信息，只在环境不接受动作时记录违规
        
        Args:
            step: 当前步数
            info: 包含详细信息的字典
        """
        self.total_steps += 1
        
        # 检查是否有环境反馈信息
        environment_acceptance = info.get("environment_acceptance", None)
        
        # 只有在环境不接受动作时才记录违规
        if environment_acceptance is False:
            self.violation_count += 1
            self.current_episode_violations += 1
            
            # 提取关键信息
            selected_action = info.get("selected_action", -1)
            selected_action_name = info.get("selected_action_name", "Unknown")
            action_is_legal = info.get("action_is_legal", False)
            environment_legal_actions = info.get("environment_legal_actions", [])
            
            # 确定违规类型
            violation_type = "environment_rejection"
            if not action_is_legal:
                violation_type = "illegal_action_selection"
            elif action_is_legal and not environment_acceptance:
                violation_type = "petri_net_constraint_violation"
            
            # 构建详细违规信息
            log_entry = {
                "step": step,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "violation_type": violation_type,
                "action": {
                    "selected": selected_action,
                    "selected_name": selected_action_name,
                    "is_legal": action_is_legal
                },
                "environment": {
                    "acceptance": environment_acceptance,
                    "legal_actions": environment_legal_actions,
                    "legal_actions_count": len(environment_legal_actions)
                },
                "state": {
                    "mask": info.get("mask", []),
                    "q_values": info.get("q_values", []),
                    "masked_q_values": info.get("masked_q_values", []),
                    "exploration_type": info.get("exploration_type", "unknown")
                }
            }
            
            # 写入JSON格式的日志条目
            self.log_file.write(json.dumps(log_entry, ensure_ascii=False))
            self.log_file.write("\n")
            self.log_file.flush()
            
            # 输出到控制台（简化版）
            print(f"[违规] Step {step}: {violation_type} - 动作 {selected_action_name}")
    
    def log_reward(self, step, reward):
        """
        记录奖励值，用于生成奖励曲线
        
        Args:
            step: 当前步数
            reward: 奖励值
        """
        self.current_episode_reward += reward
    
    def log_episode_end(self, episode, steps, terminated=False, truncated=False):
        """
        记录episode结束信息
        
        Args:
            episode: episode编号
            steps: 总步数
            terminated: 是否因终止条件结束
            truncated: 是否因截断条件结束
        """
        self.episode_count += 1
        
        # 记录episode奖励
        self.episode_rewards.append({
            "episode": episode,
            "total_reward": self.current_episode_reward,
            "steps": steps,
            "violations": self.current_episode_violations,
            "terminated": terminated,
            "truncated": truncated,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        })
        
        # 只记录有违规的episode
        if self.current_episode_violations > 0:
            log_entry = {
                "episode_end": True,
                "episode": episode,
                "total_reward": self.current_episode_reward,
                "steps": steps,
                "violations_in_episode": self.current_episode_violations,
                "terminated": terminated,
                "truncated": truncated,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }
            
            self.log_file.write(json.dumps(log_entry, ensure_ascii=False))
            self.log_file.write("\n")
            self.log_file.flush()
        
        # 重置当前episode计数器
        self.current_episode_reward = 0.0
        self.current_episode_violations = 0
    
    def get_reward_data(self):
        """获取奖励数据，用于生成奖励曲线"""
        return self.episode_rewards
    
    def get_summary(self):
        """获取违规统计摘要"""
        violation_rate = (self.violation_count / self.total_steps * 100) if self.total_steps > 0 else 0
        episodes_with_violations = sum(1 for ep in self.episode_rewards if ep["violations"] > 0)
        episodes_with_violations_rate = (episodes_with_violations / self.episode_count * 100) if self.episode_count > 0 else 0
        
        summary = {
            "total_steps": self.total_steps,
            "total_episodes": self.episode_count,
            "total_violations": self.violation_count,
            "violation_rate_percent": violation_rate,
            "violations_per_episode": self.violation_count / self.episode_count if self.episode_count > 0 else 0,
            "episodes_with_violations": episodes_with_violations,
            "episodes_with_violations_rate_percent": episodes_with_violations_rate,
            "average_reward_per_episode": np.mean([ep["total_reward"] for ep in self.episode_rewards]) if self.episode_rewards else 0,
            "average_steps_per_episode": np.mean([ep["steps"] for ep in self.episode_rewards]) if self.episode_rewards else 0
        }
        
        return summary
    
    def close(self):
        """关闭日志文件"""
        if not self.log_file.closed:
            # 写入统计摘要
            summary = self.get_summary()
            self.log_file.write("\n# VIOLATION SUMMARY\n")
            self.log_file.write(json.dumps(summary, indent=2, ensure_ascii=False))
            self.log_file.write("\n")
            
            # 写入奖励数据
            self.log_file.write("\n# REWARD DATA\n")
            self.log_file.write(json.dumps(self.episode_rewards, indent=2, ensure_ascii=False))
            self.log_file.write("\n")
            
            # 写入日志文件尾
            footer = {
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }
            self.log_file.write("\n# LOG FOOTER\n")
            self.log_file.write(json.dumps(footer, indent=2, ensure_ascii=False))
            self.log_file.close()
            
            # 输出最终统计
            print(f"\n=== 违规统计 ===")
            print(f"总步数: {summary['total_steps']}")
            print(f"总违规次数: {summary['total_violations']}")
            print(f"违规率: {summary['violation_rate_percent']:.2f}%")
            print(f"平均每episode违规: {summary['violations_per_episode']:.2f}")
            print(f"有违规的episode比例: {summary['episodes_with_violations_rate_percent']:.2f}%")
            print(f"平均每episode奖励: {summary['average_reward_per_episode']:.2f}")
    
    def get_log_file_name(self):
        """获取日志文件名"""
        return self.log_file_name