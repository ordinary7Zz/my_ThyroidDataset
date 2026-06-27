import json
import argparse
from pathlib import Path
from typing import List, Dict

def load_json_file(json_path: str) -> List[Dict]:
    """加载JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(json_path: str, data: List[Dict]):
    """保存JSON文件"""
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def merge_json_files(source_file: str, target_file: str, output_file: str = None):
    """
    将一个JSON文件的数据追加到另一个JSON文件（不去重）
    
    Args:
        source_file: 源JSON文件路径（要追加的数据）
        target_file: 目标JSON文件路径（基础数据）
        output_file: 输出JSON文件路径（如果为None，则覆盖target_file）
    """
    # 加载两个JSON文件
    print(f"正在加载目标文件: {target_file}")
    target_data = load_json_file(target_file)
    print(f"  加载了 {len(target_data)} 条记录")
    
    print(f"正在加载源文件: {source_file}")
    source_data = load_json_file(source_file)
    print(f"  加载了 {len(source_data)} 条记录")
    
    # 合并数据（不去重）
    merged_data = target_data + source_data
    print(f"\n合并后共有 {len(merged_data)} 条记录（未去重）")
    
    # 确定输出文件路径
    if output_file is None:
        output_file = target_file
        print(f"\n将覆盖目标文件: {output_file}")
    else:
        print(f"\n将保存到: {output_file}")
    
    # 保存结果
    save_json_file(output_file, merged_data)
    print(f"完成! 已保存 {len(merged_data)} 条记录到 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将一个JSON文件的数据追加到另一个JSON文件')
    parser.add_argument('source_file', type=str, 
                       help='源JSON文件路径（要追加的数据）')
    parser.add_argument('target_file', type=str,
                       help='目标JSON文件路径（基础数据）')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='输出JSON文件路径（如果未指定，则覆盖target_file）')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.source_file).exists():
        print(f"错误: 源文件不存在: {args.source_file}")
        exit(1)
    
    if not Path(args.target_file).exists():
        print(f"错误: 目标文件不存在: {args.target_file}")
        exit(1)
    
    merge_json_files(
        source_file=args.source_file,
        target_file=args.target_file,
        output_file=args.output
    )

