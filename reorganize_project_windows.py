#!/usr/bin/env python3
"""
Windows版本的Python重构脚本
更可靠、更安全的目录重构方案
"""

import os
import shutil
import glob

def create_directories():
    """创建新的目录结构"""
    print("创建新的目录结构...")
    
    directories = [
        'src/agents',
        'src/environment', 
        'src/utils',
        'src/rewards',
        'experiments/configs',
        'experiments/results',
        'experiments/logs',
        'experiments/checkpoints',
        'models/cdqn',
        'models/dqn',
        'models/gail',
        'scripts',
        'docs',
        'data'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")

def move_files():
    """移动文件到新目录结构"""
    print("移动文件...")
    
    # 1. 移动源代码
    if os.path.exists('agents'):
        for file in glob.glob('agents/*.py'):
            shutil.move(file, f'src/agents/{os.path.basename(file)}')
        try:
            os.rmdir('agents')
            print("删除空目录: agents")
        except OSError:
            pass
    
    if os.path.exists('environment'):
        for file in glob.glob('environment/*.py'):
            shutil.move(file, f'src/environment/{os.path.basename(file)}')
        try:
            os.rmdir('environment')
            print("删除空目录: environment")
        except OSError:
            pass
    
    if os.path.exists('utils'):
        # 移动 utils/petri_net
        if os.path.exists('utils/petri_net'):
            for file in glob.glob('utils/petri_net/*.py'):
                shutil.move(file, f'src/utils/{os.path.basename(file)}')
            try:
                os.rmdir('utils/petri_net')
                print("删除空目录: utils/petri_net")
            except OSError:
                pass
        
        # 移动 utils 根目录的文件
        for file in glob.glob('utils/*.py'):
            dest_file = f'src/utils/{os.path.basename(file)}'
            if not os.path.exists(dest_file):
                shutil.move(file, dest_file)
        
        try:
            os.rmdir('utils')
            print("删除空目录: utils")
        except OSError:
            pass
    
    if os.path.exists('rewards'):
        for file in glob.glob('rewards/*.py'):
            shutil.move(file, f'src/rewards/{os.path.basename(file)}')
        try:
            os.rmdir('rewards')
            print("删除空目录: rewards")
        except OSError:
            pass
    
    # 2. 移动模型文件
    if os.path.exists('lido-run-events'):
        # 移动 CDQN 模型
        for file in glob.glob('lido-run-events/*_cdqn_*.pt'):
            shutil.move(file, f'models/cdqn/{os.path.basename(file)}')
        for file in glob.glob('lido-run-events/*_cdqn_best.pt'):
            shutil.move(file, f'models/cdqn/{os.path.basename(file)}')
        
        # 移动 DQN 模型
        for file in glob.glob('lido-run-events/*_dqn_*.pt'):
            shutil.move(file, f'models/dqn/{os.path.basename(file)}')
        for file in glob.glob('lido-run-events/*_dqn_best.pt'):
            shutil.move(file, f'models/dqn/{os.path.basename(file)}')
        
        # 移动 GAIL 模型
        for file in glob.glob('lido-run-events/*gail*.pt'):
            shutil.move(file, f'models/gail/{os.path.basename(file)}')
        
        # 移动 checkpoint 文件
        for item in os.listdir('lido-run-events'):
            item_path = os.path.join('lido-run-events', item)
            if os.path.isdir(item_path) and 'checkpoints' in item:
                dest_path = f'experiments/checkpoints/{item}'
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path)
                shutil.move(item_path, dest_path)
        
        print("模型文件整理完成")
    
    # 3. 移动日志文件
    for log_dir in ['detailed_logs', 'debug_logs']:
        if os.path.exists(log_dir):
            for file in glob.glob(f'{log_dir}/*.log'):
                shutil.move(file, f'experiments/logs/{os.path.basename(file)}')
            try:
                os.rmdir(log_dir)
                print(f"删除空目录: {log_dir}")
            except OSError:
                pass
    
    # 4. 移动结果文件
    for file in glob.glob('*.csv'):
        shutil.move(file, f'experiments/results/{os.path.basename(file)}')
    
    for file in glob.glob('*.png'):
        shutil.move(file, f'experiments/results/{os.path.basename(file)}')
    
    for file in glob.glob('*.mp4'):
        shutil.move(file, f'experiments/results/{os.path.basename(file)}')
    
    # 5. 移动脚本文件
    for file in glob.glob('*.sh'):
        shutil.move(file, f'scripts/{os.path.basename(file)}')
    
    # 移动特定的 Python 脚本
    script_files = [
        'add_to_comparison.py',
        'baseline.py', 
        'evaluation.py',
        'simulation.py',
        'train.py',
        'visual.py'
    ]
    
    for file in script_files:
        if os.path.exists(file):
            shutil.move(file, f'scripts/{file}')
    
    # 移动匹配模式的脚本
    for pattern in ['verify_*.py', 'monitor_*.py', 'plot_*.py', 'read_*.py']:
        for file in glob.glob(pattern):
            shutil.move(file, f'scripts/{os.path.basename(file)}')
    
    # 6. 移动文档文件
    for file in glob.glob('*.md'):
        shutil.move(file, f'docs/{os.path.basename(file)}')
    
    # 7. 移动其他目录
    for dir_name in ['plotting', 'slurm']:
        if os.path.exists(dir_name):
            for file in glob.glob(f'{dir_name}/*'):
                shutil.move(file, f'scripts/{os.path.basename(file)}')
            try:
                os.rmdir(dir_name)
                print(f"删除空目录: {dir_name}")
            except OSError:
                pass
    
    # 8. 清理特殊目录
    if os.path.exists('.ipynb_checkpoints'):
        shutil.rmtree('.ipynb_checkpoints')
        print("删除 .ipynb_checkpoints 目录")

def create_gitignore():
    """创建 .gitignore 文件"""
    print("创建 .gitignore 文件...")
    
    gitignore_content = """# 运行产物
experiments/
models/
*.pt
*.csv
*.log
*.mp4
*.png

# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# 系统
.DS_Store
Thumbs.db
"""
    
    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    
    print(".gitignore 文件创建完成")

def show_structure():
    """显示新的目录结构"""
    print("\n新的目录结构：")
    for root, dirs, files in os.walk('.'):
        # 只显示一级目录
        level = root.replace('.', '').count(os.sep)
        if level <= 1:
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # 只显示前5个文件
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... 还有 {len(files) - 5} 个文件")

def main():
    print("开始重构项目目录结构...")
    
    create_directories()
    move_files()
    create_gitignore()
    show_structure()
    
    print("\n重构完成！")
    print("\n下一步操作：")
    print("1. 运行 update_paths_windows.py 更新路径引用")
    print("2. 测试重构后的功能是否正常")
    print("3. 提交重构到版本控制")

if __name__ == "__main__":
    main()