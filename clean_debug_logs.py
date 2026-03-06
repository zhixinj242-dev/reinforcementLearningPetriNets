#!/usr/bin/env python3
"""
【文件角色】：清理调试日志文件。
删除或压缩大型调试日志文件，释放磁盘空间。
"""
import os
import shutil
import argparse
from datetime import datetime


def clean_debug_logs(debug_dir="debug_logs", compress=False, delete=False):
    """清理调试日志文件
    
    Args:
        debug_dir: 调试日志目录
        compress: 是否压缩而不是删除
        delete: 是否删除文件
    """
    if not os.path.exists(debug_dir):
        print(f"调试日志目录 {debug_dir} 不存在")
        return
    
    total_size = 0
    file_count = 0
    
    for root, dirs, files in os.walk(debug_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            file_count += 1
            
            if delete:
                os.remove(file_path)
                print(f"已删除: {file_path} ({file_size:,} bytes)")
            elif compress:
                # 压缩文件
                zip_path = file_path + ".zip"
                shutil.make_archive(zip_path[:-4], 'zip', root_dir=os.path.dirname(file_path), 
                                 base_name=os.path.basename(file_path))
                print(f"已压缩: {file_path} -> {zip_path}")
                if os.path.exists(zip_path):
                    os.remove(file_path)
    
    if delete:
        print(f"\n已删除 {file_count} 个调试日志文件，释放 {total_size:,} bytes 空间")
    elif compress:
        print(f"\n已压缩 {file_count} 个调试日志文件")
    else:
        print(f"\n调试日志目录 {debug_dir} 包含 {file_count} 个文件，总计 {total_size:,} bytes")
        
        # 列出最大的文件
        files_with_sizes = []
        for root, dirs, files in os.walk(debug_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                files_with_sizes.append((file_path, file_size))
        
        files_with_sizes.sort(key=lambda x: x[1], reverse=True)
        
        print("\n最大的文件:")
        for file_path, file_size in files_with_sizes[:5]:
            print(f"  {file_path}: {file_size:,} bytes")


def list_debug_logs(debug_dir="debug_logs"):
    """列出调试日志文件"""
    if not os.path.exists(debug_dir):
        print(f"调试日志目录 {debug_dir} 不存在")
        return
    
    print(f"\n调试日志目录 {debug_dir} 内容:")
    print("-" * 60)
    
    for root, dirs, files in os.walk(debug_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            print(f"{file_path:40} {file_size:>10,} bytes  {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="清理调试日志文件")
    parser.add_argument("--debug-dir", default="debug_logs", help="调试日志目录")
    parser.add_argument("--list", action="store_true", help="列出调试日志文件")
    parser.add_argument("--compress", action="store_true", help="压缩而不是删除文件")
    parser.add_argument("--delete", action="store_true", help="删除调试日志文件")
    
    args = parser.parse_args()
    
    if args.list:
        list_debug_logs(args.debug_dir)
    elif args.delete:
        confirm = input(f"确定要删除 {args.debug_dir} 中的所有调试日志吗？(y/N): ")
        if confirm.lower() == 'y':
            clean_debug_logs(args.debug_dir, compress=False, delete=True)
        else:
            print("操作已取消")
    elif args.compress:
        clean_debug_logs(args.debug_dir, compress=True, delete=False)
    else:
        # 默认只显示信息
        clean_debug_logs(args.debug_dir, compress=False, delete=False)


if __name__ == "__main__":
    main()