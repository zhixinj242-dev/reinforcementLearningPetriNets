"""
【文件角色】：优化的违规日志记录器。
只在环境不接受动作时记录违规，大幅减少日志量。
"""
import json
import time
import os


class OptimizedViolationLogger:
    """优化的违规日志记录器，只记录环境拒绝的动作"""
    
    def __init__(self, log_dir="detailed_logs", algorithm_type="DQN", reward_params=None):
        """初始化违规日志记录器
        
        Args:
            log_dir: 日志目录
            algorithm_type: 算法类型 (DQN/CDQN)
            reward_params: 奖励参数
        """
        self.log_dir = log_dir
        self.algorithm_type = algorithm_type
        self.reward_params = reward_params or {}
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成日志文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        reward_str = "_".join([f"{k}{v}" for k, v in self.reward_params.items()])
        self.log_filename = f"{algorithm_type}_{reward_str}_{timestamp}.jsonl"
        self.log_filepath = os.path.join(log_dir, self.log_filename)
        
        # 打开日志文件
        self.log_file = open(self.log_filepath, 'w', encoding='utf-8')
        
        # 统计信息
        self.total_steps = 0
        self.violation_count = 0
        self.current_episode_violations = 0
        
        print(f"[OptimizedViolationLogger] 日志文件: {self.log_filepath}")
    
    def log_step(self, step, info):
        """记录步骤信息，只在环境不接受动作时记录违规
        
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
                    "masked_q_values": info.get("masked_q_values", [])
                }
            }
            
            # 写入JSON格式的日志条目
            self.log_file.write(json.dumps(log_entry, ensure_ascii=False))
            self.log_file.write("\n")
            self.log_file.flush()
            
            # 输出到控制台（简化版）- 减少输出
            # print(f"[违规] Step {step}: {violation_type} - 动作 {selected_action_name}")
    
    def reset_episode(self):
        """重置episode统计"""
        self.current_episode_violations = 0
    
    def get_stats(self):
        """获取统计信息"""
        return {
            "total_steps": self.total_steps,
            "violation_count": self.violation_count,
            "violation_rate": self.violation_count / max(self.total_steps, 1)
        }
    
    def close(self):
        """关闭日志文件"""
        if self.log_file:
            self.log_file.close()
            # print(f"[OptimizedViolationLogger] 日志已保存: {self.log_filepath}")  # 减少输出
            # print(f"[OptimizedViolationLogger] 统计: {self.get_stats()}")  # 减少输出