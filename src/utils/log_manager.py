import os
import time
import json

class LogManager:
    """
    日志管理模块，统一CDQN和DQN的日志输出格式
    """
    
    def __init__(self, algorithm_type, reward_params, log_dir="detailed_logs"):
        """
        初始化日志管理器
        
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
        self.log_file_name = f"{self.log_dir}/{self.algorithm_type}_{params_str}_{timestamp}.log"
        
        # 打开日志文件
        self.log_file = open(self.log_file_name, "w", encoding="utf-8")
        
        # 写入日志文件头
        self.write_header()
    
    def write_header(self):
        """
        写入日志文件头，包含算法类型和奖励函数参数
        """
        header = {
            "algorithm_type": self.algorithm_type,
            "reward_params": self.reward_params,
            "log_file_name": self.log_file_name,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        self.log_file.write("# LOG HEADER\n")
        self.log_file.write(json.dumps(header, indent=2, ensure_ascii=False))
        self.log_file.write("\n\n# STEP LOGS\n")
        self.log_file.flush()
    
    def _convert_numpy_to_list(self, obj):
        """
        将包含NumPy数组的对象转换为包含Python列表的对象，以便JSON序列化
        
        Args:
            obj: 要转换的对象
            
        Returns:
            转换后的对象
        """
        import numpy as np
        if isinstance(obj, dict):
            return {k: self._convert_numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [self._convert_numpy_to_list(item) for item in obj]
        else:
            return obj

    def log_step(self, step, info):
        """
        记录每一步的详细信息
        
        Args:
            step: 当前步数
            info: 包含详细信息的字典
        """
        # 转换NumPy数组为Python列表，以便JSON序列化
        converted_info = self._convert_numpy_to_list(info)
        
        log_entry = {
            "step": step,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "algorithm_type": self.algorithm_type,
            "info": converted_info
        }
        
        # 写入JSON格式的日志条目
        self.log_file.write(json.dumps(log_entry, ensure_ascii=False))
        self.log_file.write("\n")
        self.log_file.flush()
    
    def update_step(self, step, info):
        """
        更新现有的日志条目
        
        Args:
            step: 当前步数
            info: 包含详细信息的字典
        """
        # 转换NumPy数组为Python列表，以便JSON序列化
        converted_info = self._convert_numpy_to_list(info)
        
        # 由于日志文件是顺序写入的，我们无法直接修改之前的条目
        # 所以我们创建一个新的条目，标记为更新
        log_entry = {
            "step": step,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "algorithm_type": self.algorithm_type,
            "info": converted_info,
            "type": "update"
        }
        
        # 写入JSON格式的日志条目
        self.log_file.write(json.dumps(log_entry, ensure_ascii=False))
        self.log_file.write("\n")
        self.log_file.flush()
    
    def close(self):
        """
        关闭日志文件
        """
        if not self.log_file.closed:
            # 写入日志文件尾
            footer = {
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }
            self.log_file.write("\n# LOG FOOTER\n")
            self.log_file.write(json.dumps(footer, indent=2, ensure_ascii=False))
            self.log_file.close()
    
    def get_log_file_name(self):
        """
        获取日志文件名
        
        Returns:
            日志文件名
        """
        return self.log_file_name
