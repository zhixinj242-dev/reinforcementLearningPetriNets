# Windows 项目重构指南

## 重构步骤

### 1. 执行目录重构
```cmd
# 在项目根目录执行
reorganize_project_windows.bat
```

### 2. 更新路径引用
```cmd
# 执行路径更新脚本
python update_paths_windows.py
```

### 3. 验证重构结果
```cmd
# 检查新的目录结构
tree /F

# 检查文件是否正确移动
dir src\
dir scripts\
dir models\
dir experiments\
```

### 4. 测试功能
```cmd
# 测试训练功能
cd scripts
python train.py --help

# 测试评估功能
python evaluation.py --help

# 测试仿真功能
python simulation.py --help
```

## 重构后的目录结构

```
reinforcementLearningPetriNets\
├── src\                    # 源代码
│   ├── agents\            # CDQN, DQN, GAIL实现
│   ├── environment\        # Petri网环境
│   ├── utils\             # 工具模块
│   └── rewards\           # 奖励函数
├── experiments\            # 实验相关
│   ├── results\           # CSV结果文件
│   ├── logs\              # 详细日志
│   └── checkpoints\        # 检查点文件
├── models\                 # 训练模型
│   ├── cdqn\             # CDQN模型文件
│   ├── dqn\              # DQN模型文件
│   └── gail\             # GAIL模型文件
├── scripts\                # 所有运行脚本
├── data\                  # Petri网数据文件
└── docs\                  # 文档和说明
```

## Windows 环境的特殊处理

### 1. 路径分隔符
- 使用反斜杠 `\` 作为路径分隔符
- 双反斜杠 `\\` 用于字符串中的路径
- 路径拼接使用 `os.path.join()` 或原始字符串

### 2. 文件移动命令
- 使用 `move /Y` 命令移动文件
- 使用 `xcopy /E /I /Y` 复制目录
- 使用 `rmdir /S /Q` 删除目录

### 3. 批处理文件处理
- 使用 `for %%f in (*.*)` 循环处理文件
- 使用 `2>nul` 抑制错误输出

## 主要改进

### 1. 目录功能分离
- **src\**: 纯源代码，便于版本控制
- **experiments\**: 所有实验相关文件统一管理
- **models\**: 按算法类型分类存储模型
- **scripts\**: 所有可执行脚本集中管理

### 2. 命名规范
- 模型文件: `cdqn_s1.0_c0.0_w1.0_best.pt`
- 日志文件: `cdqn_s1.0_c0.0_w1.0_20260205_122715.log`
- 结果文件: `method_comparison.csv`, `simulation_results.csv`

### 3. 路径统一
- 所有脚本使用相对路径
- 统一的数据输入输出路径
- 适配 Windows 路径格式

## 注意事项

1. **备份重要数据**: 重构前确保重要数据已备份
2. **测试功能**: 重构后测试所有主要功能
3. **逐步迁移**: 不要一次性修改太多，分步骤验证
4. **版本控制**: 重构完成后及时提交到版本控制

## 故障排除

### 如果文件移动失败
```cmd
REM 手动移动特定文件
move agents\dqn.py src\agents\
move environment\petri_net.py src\environment\
```

### 如果路径更新失败
```cmd
REM 手动编辑脚本文件
REM 查找需要更新的路径
findstr /n "lido-run-events" scripts\*.py
```

### 如果功能测试失败
```cmd
REM 检查Python路径
set PYTHONPATH=%PYTHONPATH%;%CD%\src
python scripts\train.py --help
```

### 如果批处理文件无法执行
```cmd
REM 确保在正确的目录
cd /d "python练习\reinforcementLearningPetriNets"
reorganize_project_windows.bat
```

## 重构后的好处

1. **结构清晰**: 功能模块化，易于理解和维护
2. **版本控制友好**: 源代码和运行产物分离
3. **协作便利**: 其他人容易理解项目结构
4. **扩展性好**: 新功能容易添加到对应模块
5. **部署简单**: 只需部署必要的文件和目录
6. **Windows兼容**: 完全适配 Windows 环境的路径和命令

## 验证清单

重构完成后，请验证以下项目：

- [ ] 所有源代码文件已移动到 `src\` 目录
- [ ] 所有脚本文件已移动到 `scripts\` 目录
- [ ] 所有模型文件已分类移动到 `models\` 目录
- [ ] 所有日志文件已移动到 `experiments\logs\` 目录
- [ ] 所有结果文件已移动到 `experiments\results\` 目录
- [ ] `.gitignore` 文件已创建
- [ ] 路径引用已更新
- [ ] 主要功能可以正常运行