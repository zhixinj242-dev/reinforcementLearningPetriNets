#!/usr/bin/env python3
# 违规跟踪使用示例

from utils.violation_logger import ViolationLogger

# 创建违规日志记录器
violation_logger = ViolationLogger("CDQN", {
    'success': 1.0,
    'waiting_time': 1.0,
    'max_waiting_time': 1.5
})

# 在训练循环中使用
for episode in range(num_episodes):
    episode_violations = 0
    
    for step in range(max_steps):
        # 执行动作
        action, reward, done, info = env.step(action)
        
        # 检查是否有违规
        if info.get('violation', False):
            episode_violations += 1
            violation_logger.log_violation(step, episode, {
                'violation_type': info.get('violation_type', 'unknown'),
                'action': action,
                'state': info.get('state', 'unknown'),
                'reason': info.get('violation_reason', 'unknown')
            })
        
        if done:
            break
    
    # 记录episode结束
    violation_logger.log_episode_end(episode, total_reward, step, episode_violations)

# 关闭日志记录器
violation_logger.close()
