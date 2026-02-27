import torch

def verify_model_fix():
    print("=== 验证 Model 层 Shift-then-Mask 逻辑 ===")
    
    # 模拟网络原始输出 Q 值 (Batch Size = 2)
    # Case 1: 正常情况，合法动作 Q 值较高
    # Case 2: "负 Q 值陷阱"，非法动作 Q 值(-5) 比合法动作(-50) 高，且都是负数
    output = torch.tensor([
        [10.0, 5.0, -10.0, 0.0],   # Case 1
        [-5.0, -50.0, -100.0, -5.0] # Case 2
    ])
    
    # 模拟 Mask (1=合法, 0=非法)
    # Case 1: 动作 0,1 合法
    # Case 2: 只有动作 1 合法 (注意动作 0 是 -5 但非法)
    mask = torch.tensor([
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ])
    
    print(f"原始 Output:\n{output}")
    print(f"Mask:\n{mask}")
    
    # --- 模拟 ConstrainedModel.compute 中的逻辑 ---
    
    # 1. 动态平移：找出当前 Batch 中最小的 Q 值
    min_q = torch.min(output)
    offset = 0.0
    if min_q < 0:
        offset = torch.abs(min_q) + 1.0
    
    print(f"\n计算出的 Offset: {offset}")
    
    # 2. 平移 -> 乘掩码 -> 还原
    shifted_output = output + offset
    masked_shifted_output = shifted_output * mask
    final_output = masked_shifted_output - offset
    
    print(f"\nFinal Output (Model 输出):\n{final_output}")
    
    # --- 验证结果 ---
    
    # Case 2 分析
    case2_out = final_output[1]
    print(f"\nCase 2 (负值陷阱) 详细分析:")
    print(f"  动作 0 (非法, 原值 -5.0): 现值 {case2_out[0]}")
    print(f"  动作 1 (合法, 原值 -50.0): 现值 {case2_out[1]}")
    
    # 验证 argmax 是否正确
    action = torch.argmax(case2_out).item()
    print(f"  Argmax 选择的动作: {action}")
    
    if action == 1:
        print("  ✅ 成功：正确选择了唯一的合法动作 1")
    else:
        print("  ❌ 失败：选择了非法动作")
        
    # 验证合法动作的值是否保持原样
    if case2_out[1] == -50.0:
         print("  ✅ 成功：合法动作的 Q 值保持不变")
    else:
         print("  ❌ 失败：合法动作的 Q 值被改变了")

if __name__ == "__main__":
    verify_model_fix()
