#!/usr/bin/env python3
"""
测试RewardLogger导入
"""
try:
    from utils.reward_logger import RewardLogger
    print("✅ RewardLogger导入成功")
    
    # 测试创建实例
    logger = RewardLogger(
        log_dir="test_logs",
        algorithm_type="CDQN",
        reward_params={"success": 1.0, "waiting_time": 1.0},
        window_size=10
    )
    print("✅ RewardLogger实例创建成功")
    
    # 测试记录
    logger.log_step(1, 5.0, {"environment_acceptance": True})
    logger.log_episode(1, 50.0, 30, 2)
    logger.close()
    print("✅ RewardLogger测试完成")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()