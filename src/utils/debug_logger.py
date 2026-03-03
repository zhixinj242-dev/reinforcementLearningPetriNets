import os
import sys
import time
import traceback
import logging
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import json
import numpy as np


class ErrorSeverity(Enum):
    """错误严重程度枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """错误类别枚举"""
    ENVIRONMENT = "environment"
    AGENT = "agent"
    TRAINING = "training"
    DATA = "data"
    SYSTEM = "system"
    IMPORT = "import"
    CONFIGURATION = "configuration"


class DebugLogger:
    """调试日志记录器，提供全面的错误处理和调试信息"""
    
    def __init__(self, log_dir="experiments/debug", log_level=logging.DEBUG):
        """
        初始化调试日志记录器
        
        Args:
            log_dir: 日志目录
            log_level: 日志级别
        """
        self.log_dir = log_dir
        self.log_level = log_level
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成日志文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.log_file = os.path.join(log_dir, f"debug_{timestamp}.log")
        
        # 设置日志格式
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("RLPN_Debug")
        
        # 错误统计
        self.error_counts = {}
        self.error_details = []
        
        # 系统信息
        self.system_info = self._collect_system_info()
        
        # 记录初始化信息
        self.logger.info("调试日志记录器已初始化")
        self.logger.info(f"日志文件: {self.log_file}")
        self.logger.info(f"系统信息: {json.dumps(self.system_info, indent=2)}")
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """收集系统信息"""
        try:
            import platform
            import psutil
            
            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": os.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_usage": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent,
                "environment_variables": {
                    "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                    "PATH": os.environ.get("PATH", "")[:200] + "..." if len(os.environ.get("PATH", "")) > 200 else os.environ.get("PATH", "")
                }
            }
        except Exception as e:
            self.logger.warning(f"收集系统信息失败: {e}")
            return {"error": str(e)}
    
    def log_error(self, error: Exception, category: ErrorCategory, severity: ErrorSeverity, 
                  context: Dict[str, Any] = None, additional_info: Dict[str, Any] = None):
        """
        记录错误信息
        
        Args:
            error: 异常对象
            category: 错误类别
            severity: 错误严重程度
            context: 错误上下文
            additional_info: 额外信息
        """
        # 构建错误信息
        error_info = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "category": category.value,
            "severity": severity.value,
            "context": context or {},
            "additional_info": additional_info or {},
            "traceback": traceback.format_exc()
        }
        
        # 记录到日志
        self.logger.error(f"[{severity.value.upper()}] {category.value}: {error_info['error_type']}: {error_info['error_message']}")
        if context:
            self.logger.error(f"Context: {json.dumps(context, indent=2)}")
        if additional_info:
            self.logger.error(f"Additional Info: {json.dumps(additional_info, indent=2)}")
        
        # 记录详细错误信息
        self.logger.debug(f"Full traceback:\n{error_info['traceback']}")
        
        # 更新错误统计
        error_key = f"{category.value}_{error_info['error_type']}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        self.error_details.append(error_info)
        
        # 如果是严重错误，立即保存错误信息
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.save_error_summary()
    
    def log_warning(self, message: str, context: Dict[str, Any] = None):
        """记录警告信息"""
        self.logger.warning(message)
        if context:
            self.logger.warning(f"Context: {json.dumps(context, indent=2)}")
    
    def log_info(self, message: str, context: Dict[str, Any] = None):
        """记录信息"""
        self.logger.info(message)
        if context:
            self.logger.debug(f"Context: {json.dumps(context, indent=2)}")
    
    def log_debug(self, message: str, context: Dict[str, Any] = None):
        """记录调试信息"""
        self.logger.debug(message)
        if context:
            self.logger.debug(f"Context: {json.dumps(context, indent=2)}")
    
    def check_environment(self) -> Dict[str, Any]:
        """检查环境状态"""
        env_status = {
            "python_path": sys.path,
            "working_directory": os.getcwd(),
            "file_permissions": {},
            "module_imports": {}
        }
        
        # 检查关键文件权限
        key_files = [
            "src/agents/__init__.py",
            "src/environment/__init__.py",
            "src/utils/__init__.py",
            "src/rewards/__init__.py",
            "data/traffic-scenario.PNPRO"
        ]
        
        for file_path in key_files:
            full_path = os.path.join(os.getcwd(), file_path)
            env_status["file_permissions"][file_path] = {
                "exists": os.path.exists(full_path),
                "readable": os.access(full_path, os.R_OK) if os.path.exists(full_path) else False,
                "writable": os.access(full_path, os.W_OK) if os.path.exists(full_path) else False
            }
        
        # 检查关键模块导入
        key_modules = [
            "agents.dqn",
            "environment",
            "utils.parser_pnpro",
            "utils.optimized_violation_logger",
            "utils.reward_tracker",
            "rewards"
        ]
        
        for module_name in key_modules:
            try:
                __import__(module_name)
                env_status["module_imports"][module_name] = "success"
            except ImportError as e:
                env_status["module_imports"][module_name] = f"failed: {str(e)}"
        
        self.logger.info(f"环境检查完成: {json.dumps(env_status, indent=2)}")
        return env_status
    
    def check_training_state(self, agent, env, trainer) -> Dict[str, Any]:
        """检查训练状态"""
        training_state = {
            "agent": {
                "type": type(agent).__name__,
                "device": str(agent.device) if hasattr(agent, 'device') else "unknown",
                "memory_size": agent.memory.memory_size if hasattr(agent, 'memory') else "unknown"
            },
            "environment": {
                "type": type(env).__name__,
                "action_space": str(env.action_space) if hasattr(env, 'action_space') else "unknown",
                "observation_space": str(env.observation_space) if hasattr(env, 'observation_space') else "unknown"
            },
            "trainer": {
                "type": type(trainer).__name__,
                "timesteps": trainer.timesteps if hasattr(trainer, 'timesteps') else "unknown"
            }
        }
        
        self.logger.info(f"训练状态检查: {json.dumps(training_state, indent=2)}")
        return training_state
    
    def save_error_summary(self):
        """保存错误摘要"""
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "system_info": self.system_info,
            "error_counts": self.error_counts,
            "total_errors": len(self.error_details),
            "errors_by_severity": {},
            "errors_by_category": {}
        }
        
        # 按严重程度统计
        for severity in ErrorSeverity:
            summary["errors_by_severity"][severity.value] = sum(
                1 for error in self.error_details if error["severity"] == severity.value
            )
        
        # 按类别统计
        for category in ErrorCategory:
            summary["errors_by_category"][category.value] = sum(
                1 for error in self.error_details if error["category"] == category.value
            )
        
        # 保存摘要
        summary_file = os.path.join(self.log_dir, "error_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 保存详细错误信息
        details_file = os.path.join(self.log_dir, "error_details.json")
        with open(details_file, 'w') as f:
            json.dump(self.error_details, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"错误摘要已保存到: {summary_file}")
        self.logger.info(f"错误详情已保存到: {details_file}")
    
    def close(self):
        """关闭调试日志记录器"""
        self.save_error_summary()
        self.logger.info("调试日志记录器已关闭")


class ErrorRecovery:
    """错误恢复机制"""
    
    def __init__(self, debug_logger: DebugLogger):
        self.debug_logger = debug_logger
        self.recovery_strategies = {
            ErrorCategory.IMPORT: self._recover_import_error,
            ErrorCategory.ENVIRONMENT: self._recover_environment_error,
            ErrorCategory.AGENT: self._recover_agent_error,
            ErrorCategory.TRAINING: self._recover_training_error,
            ErrorCategory.DATA: self._recover_data_error,
            ErrorCategory.SYSTEM: self._recover_system_error,
            ErrorCategory.CONFIGURATION: self._recover_configuration_error
        }
    
    def handle_error(self, error: Exception, category: ErrorCategory, 
                    context: Dict[str, Any] = None) -> Tuple[bool, str]:
        """
        处理错误并尝试恢复
        
        Returns:
            Tuple[bool, str]: (是否成功恢复, 恢复消息)
        """
        try:
            recovery_func = self.recovery_strategies.get(category)
            if recovery_func:
                return recovery_func(error, context or {})
            else:
                return False, f"没有针对 {category.value} 类别的恢复策略"
        except Exception as recovery_error:
            self.debug_logger.log_error(
                recovery_error, 
                ErrorCategory.SYSTEM, 
                ErrorSeverity.HIGH,
                {"original_error": str(error), "category": category.value},
                {"recovery_attempt": "failed"}
            )
            return False, f"恢复过程中出现错误: {str(recovery_error)}"
    
    def _recover_import_error(self, error: ImportError, context: Dict[str, Any]) -> Tuple[bool, str]:
        """恢复导入错误"""
        module_name = context.get("module_name", "unknown")
        
        # 尝试添加路径
        if "src" not in sys.path:
            sys.path.insert(0, os.path.join(os.getcwd(), "src"))
            self.debug_logger.log_info(f"已添加 src 目录到 Python 路径")
            return True, "通过添加路径恢复导入错误"
        
        # 尝试重新导入
        try:
            import importlib
            importlib.import_module(module_name)
            return True, f"成功重新导入模块 {module_name}"
        except ImportError:
            return False, f"无法导入模块 {module_name}"
    
    def _recover_environment_error(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, str]:
        """恢复环境错误"""
        # 尝试重置环境
        try:
            env = context.get("environment")
            if env and hasattr(env, 'reset'):
                env.reset()
                return True, "环境重置成功"
        except Exception as e:
            return False, f"环境重置失败: {str(e)}"
        
        return False, "无法恢复环境错误"
    
    def _recover_agent_error(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, str]:
        """恢复智能体错误"""
        # 尝试重新初始化智能体
        return False, "智能体错误恢复策略尚未实现"
    
    def _recover_training_error(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, str]:
        """恢复训练错误"""
        # 尝试降低学习率或调整参数
        return False, "训练错误恢复策略尚未实现"
    
    def _recover_data_error(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, str]:
        """恢复数据错误"""
        # 尝试重新加载数据
        return False, "数据错误恢复策略尚未实现"
    
    def _recover_system_error(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, str]:
        """恢复系统错误"""
        # 尝试清理资源
        try:
            import gc
            gc.collect()
            return True, "系统资源清理完成"
        except Exception:
            return False, "系统资源清理失败"
    
    def _recover_configuration_error(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, str]:
        """恢复配置错误"""
        # 尝试使用默认配置
        return False, "配置错误恢复策略尚未实现"


def create_debug_logger(log_dir="experiments/debug") -> DebugLogger:
    """创建调试日志记录器的工厂函数"""
    return DebugLogger(log_dir)


def create_error_recovery(debug_logger: DebugLogger) -> ErrorRecovery:
    """创建错误恢复机制的工厂函数"""
    return ErrorRecovery(debug_logger)