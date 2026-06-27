import os
import json
import shutil
from typing import List, Dict, Tuple


def find_single_json_file(folder: str) -> str:
    """在给定目录中找到唯一的 json 文件并返回其路径。"""
    json_files = [f for f in os.listdir(folder) if f.lower().endswith(".json")]
    if not json_files:
        raise FileNotFoundError(f"目录中未找到 json 文件: {folder}")
    if len(json_files) > 1:
        raise RuntimeError(f"目录中存在多个 json 文件，请手动指定或清理: {folder}, {json_files}")
    return os.path.join(folder, json_files[0])


def load_json_records(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"json 文件不是列表结构: {json_path}")
    return data


def save_json_records(json_path: str, records: List[Dict]) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def split_records(records: List[Dict], keep_num: int) -> Tuple[List[Dict], List[Dict]]:
    """
    按 filename 排序后，前 keep_num 条作为保留，其余作为移动。
    """
    if len(records) < keep_num:
        raise ValueError(f"记录数量({len(records)})少于要保留的数量({keep_num})")

    sorted_records = sorted(records, key=lambda x: x.get("filename", ""))
    keep = sorted_records[:keep_num]
    move = sorted_records[keep_num:]
    return keep, move


def move_file(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # 如目标已存在，这里选择覆盖；如需保守策略可改为判断并报错
    shutil.move(src, dst)


def move_image_and_masks(
    filename: str,
    valid_images_dir: str,
    valid_masks_dir: str,
    train_images_dir: str,
    train_masks_dir: str,
) -> None:
    """
    根据 image filename 移动：
    - valid/images/filename -> train/images/filename
    - valid/masks/下所有同 basename 的 mask 文件 -> train/masks/
    """
    # 移动 image
    src_img = os.path.join(valid_images_dir, filename)
    dst_img = os.path.join(train_images_dir, filename)

    if not os.path.exists(src_img):
        raise FileNotFoundError(f"在 valid images 中找不到图像文件: {src_img}")
    move_file(src_img, dst_img)

    # 移动对应的 mask（同名不同后缀也算）
    base_name, _ = os.path.splitext(filename)
    if os.path.isdir(valid_masks_dir):
        for name in os.listdir(valid_masks_dir):
            stem, ext = os.path.splitext(name)
            if stem == base_name:
                src_mask = os.path.join(valid_masks_dir, name)
                dst_mask = os.path.join(train_masks_dir, name)
                move_file(src_mask, dst_mask)


def process_one_pair(train_dir: str, valid_dir: str, keep_num: int = 100) -> None:
    print("=" * 80)
    print(f"处理这一对目录：")
    print(f"  train_dir = {train_dir}")
    print(f"  valid_dir = {valid_dir}")
    print(f"  keep_num  = {keep_num}")

    # 路径准备
    train_images_dir = os.path.join(train_dir, "images")
    train_masks_dir = os.path.join(train_dir, "masks")
    valid_images_dir = os.path.join(valid_dir, "images")
    valid_masks_dir = os.path.join(valid_dir, "masks")

    # 找 json；如果未找到 json 文件，则跳过该对目录的 json 相关处理
    try:
        train_json_path = find_single_json_file(train_dir)
        valid_json_path = find_single_json_file(valid_dir)
    except FileNotFoundError as e:
        print(f"[SKIP] {e}，跳过该对目录的 json 处理。")
        print("=" * 80)
        print()
        return
    print(f"Train json: {train_json_path}")
    print(f"Valid json: {valid_json_path}")

    # 读 json
    train_records = load_json_records(train_json_path)
    valid_records = load_json_records(valid_json_path)

    print(f"Train 现有记录数: {len(train_records)}")
    print(f"Valid 现有记录数: {len(valid_records)}")

    # 如果 valid 中数量少于 keep_num，则不做任何移动操作
    if len(valid_records) < keep_num:
        print(f"[SKIP] Valid 记录数 ({len(valid_records)}) 少于要保留的数量 ({keep_num})，不进行移动操作。")
        print("=" * 80)
        print()
        return

    # 拆分 valid 记录
    keep_records, move_records = split_records(valid_records, keep_num)
    print(f"Valid 中保留 {len(keep_records)} 条，移动 {len(move_records)} 条到 train")

    # 移动对应文件
    for rec in move_records:
        filename = rec.get("filename")
        if not filename:
            raise ValueError(f"记录中缺少 'filename' 字段: {rec}")
        move_image_and_masks(
            filename=filename,
            valid_images_dir=valid_images_dir,
            valid_masks_dir=valid_masks_dir,
            train_images_dir=train_images_dir,
            train_masks_dir=train_masks_dir,
        )

    # 更新 json
    new_train_records = train_records + move_records
    new_valid_records = keep_records

    save_json_records(train_json_path, new_train_records)
    save_json_records(valid_json_path, new_valid_records)

    # 简单检查
    print("=== 操作完成，简单检查 ===")
    print(f"Train 新记录数: {len(new_train_records)}")
    print(f"Valid 新记录数: {len(new_valid_records)}")
    if len(new_valid_records) == keep_num:
        print(f"[CHECK] ✅ Valid 中正好保留 {keep_num} 条记录。")
    else:
        print(f"[CHECK] ❌ Valid 中记录数量为 {len(new_valid_records)}，不是 {keep_num}，请检查。")
    print("=" * 80)
    print()


if __name__ == "__main__":
    # 在这里配置多组 train / valid 目录及各自要保留的数量
    # 每一项是一个字典：{"train_dir": "...", "valid_dir": "...", "keep_num": 100}
    CONFIGS = [
        {
            "train_dir": "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/train",
            "valid_dir": "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/valid",
            "keep_num": 100,
        },
        {
            "train_dir": "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/train",
            "valid_dir": "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/valid",
            "keep_num": 100,
        },
        {
            "train_dir": "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/PKTN_comprehensive/train",
            "valid_dir": "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/PKTN_comprehensive/valid",
            "keep_num": 100,
        },
        {
            "train_dir": "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/train",
            "valid_dir": "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/valid",
            "keep_num": 100,
        },
        {
            "train_dir": "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/train",
            "valid_dir": "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/valid",
            "keep_num": 100,
        },
    ]

    for cfg in CONFIGS:
        process_one_pair(
            train_dir=cfg["train_dir"],
            valid_dir=cfg["valid_dir"],
            keep_num=cfg.get("keep_num", 100),
        )