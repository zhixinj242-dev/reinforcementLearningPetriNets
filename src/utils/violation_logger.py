import os
import time
import json

class ViolationLogger:
    """
    简化的日志管理器，只记录违规动作
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
        self.log_file_name = f"{self.log_dir}/violations_{self.algorithm_type}_{params_str}_{timestamp}.log"
        
        # 打开日志文件
        self.log_file = open(self.log_file_name, "w", encoding="utf-8")
        
        # 统计信息
        self.violation_count = 0
        self.total_steps = 0
        self.episode_count = 0
        
        # 写入日志文件头
        self.write_header()
    
    def write_header(self):
        """写入日志文件头"""
        header = {
            "algorithm_type": self.algorithm_type,
            "reward_params": self.reward_params,
            "log_file_name": self.log_file_name,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        self.log_file.write("# VIOLATION LOG HEADER\n")
        self.log_file.write(json.dumps(header, indent=2, ensure_ascii=False))
        self.log_file.write("\n\n# VIOLATION LOGS\n")
        self.log_file.flush()
    
    def log_violation(self, step, episode, info):
        """
        记录违规动作
        
        Args:
            step: 当前步数
            episode: 当前episode
            info: 包含详细信息的字典
        """
        self.violation_count += 1
        
        log_entry = {
            "step": step,
            "episode": episode,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "violation_type": info.get("violation_type", "unknown"),
            "action": info.get("action", "unknown"),
            "state": info.get("state", "unknown"),
            "reason": info.get("reason", "unknown")
        }
        
        # 写入JSON格式的日志条目
        self.log_file.write(json.dumps(log_entry, ensure_ascii=False))
        self.log_file.write("\n")
        self.log_file.flush()
        
        # 同时输出到控制台
        print(f"[违规] Episode {episode}, Step {step}: {log_entry['violation_type']} - {log_entry['reason']}")
    
    def log_episode_end(self, episode, total_reward, steps, violations_in_episode):
        """
        记录episode结束信息
        
        Args:
            episode: episode编号
            total_reward: 总奖励
            steps: 总步数
            violations_in_episode: 本episode违规次数
        """
        self.episode_count += 1
        self.total_steps += steps
        
        # 只记录有违规的episode
        if violations_in_episode > 0:
            log_entry = {
                "episode_end": True,
                "episode": episode,
                "total_reward": total_reward,
                "steps": steps,
                "violations_in_episode": violations_in_episode,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }
            
            self.log_file.write(json.dumps(log_entry, ensure_ascii=False))
            self.log_file.write("\n")
            self.log_file.flush()
    
    def get_summary(self):
        """获取违规统计摘要"""
        violation_rate = (self.violation_count / self.total_steps * 100) if self.total_steps > 0 else 0
        
        summary = {
            "total_steps": self.total_steps,
            "total_episodes": self.episode_count,
            "total_violations": self.violation_count,
            "violation_rate_percent": violation_rate,
            "violations_per_episode": self.violation_count / self.episode_count if self.episode_count > 0 else 0
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
    
    def get_log_file_name(self):
        """获取日志文件名"""
        return self.log_file_name