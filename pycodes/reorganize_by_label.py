"""
================================================================================
脚本名称：reorganize_by_label.py
功能说明：将平铺存放的图像和掩码，按分类标签重新组织到带标签名的子目录中。

旧结构（输入）：
    images/                    ← 所有图像混在一个目录下
    masks/                     ← 所有掩码混在一个目录下
    label.json                 ← JSON 数组，每条记录包含 filename 和分类标签字段

新结构（输出）：
    target_dir/
    ├── images/
    │   ├── 0/                 ← 标签值为 0 的所有图像
    │   │   ├── img_001.jpg
    │   │   └── img_003.jpg
    │   ├── 1/                 ← 标签值为 1 的所有图像
    │   │   ├── img_002.jpg
    │   │   └── img_004.jpg
    │   └── ...
    └── masks/
        ├── 0/                 ← 标签值为 0 的所有掩码
        │   ├── img_001.png
        │   └── img_003.png
        ├── 1/                 ← 标签值为 1 的所有掩码
        │   ├── img_002.png
        │   └── img_004.png
        └── ...

核心逻辑：
    1. 读取 JSON 文件（数组格式），每条记录必须有 "filename" 字段和指定的分类标签字段。
    2. 根据 filename 定位源图像（filename 可能含子目录路径，如 "2016/张三/img.jpg"）。
    3. 按文件名不含扩展名的部分（stem）在 mask 目录中匹配对应掩码。
    4. 将图像和掩码分别复制到 target/images/{标签值}/ 和 target/masks/{标签值}/ 下，
       目标文件名使用原始文件名（去掉路径前缀，只保留文件名部分）。

JSON 文件格式示例（sample.json）：
    [
      {
        "filename": "2016/刘桂英/刘桂英_01_0011_0011.jpg",
        "malignancy": 1,
        "LNM_CN01": 0,
        "FTCPTC": 0,
        "tirads": 1
      },
      ...
    ]

使用方法（命令行）：
    python reorganize_by_label.py

    使用前修改 main 块中的配置变量（见 if __name__ == "__main__" 部分）。

使用方法（作为模块导入）：
    from reorganize_by_label import reorganize_by_label

    reorganize_by_label(
        src_image_dir="/data/old/images",
        src_mask_dir="/data/old/masks",
        json_path="/data/label.json",
        target_base_dir="/data/new",
        label_field="malignancy",
    )

参数说明：
    - src_image_dir : 源图像目录（image 和 mask 的文件名是一一对应的，不含后缀）
    - src_mask_dir  : 源掩码目录
    - json_path     : 分类标签 JSON 文件路径
    - target_base_dir : 目标根目录（会自动在其下创建 images/ 和 masks/ 子目录）
    - label_field   : JSON 中用作分类标签的字段名，如 "malignancy"、"tirads"

注意事项：
    - 图像和掩码通过文件名 stem（不含扩展名）匹配，大小写不敏感。
    - filename 字段可能含子目录路径，定位源文件时会拼接 src_image_dir + filename，
      但复制到目标时只保留纯文件名，不保留子目录结构。
    - 如果某条记录在 mask 目录中找不到对应掩码，会打印警告并跳过该条目。
    - 复制使用 shutil.copy2，保留文件元数据。
    - 最终会打印各类别的图像/掩码数量统计。
================================================================================
"""

import os
import shutil
import json
from typing import Dict, Iterable
from collections import defaultdict

# ==============================================================================
# 配置常量（修改下面的路径和字段名以适配你的数据）
# ==============================================================================

# 源图像目录（image 和 mask 的文件名是一一对应的，通过 stem 匹配）
SRC_IMAGE_DIR = "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/test/images"

# 源掩码目录
SRC_MASK_DIR = "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/test/masks"

# 分类标签 JSON 文件路径
JSON_PATH = "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/test/Cine-Clip_test_label.json"

# 目标根目录（会自动在其下创建 images/{标签值}/ 和 masks/{标签值}/ 子目录）
TARGET_BASE_DIR = "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/Cine-Clip_by_BM/test"

# 分类字段名（JSON 中用作分类标签的 key）
# 可选值: "malignancy", "LNM_CN01", "FTCPTC", "tirads"
LABEL_FIELD = "malignancy"


def reorganize_by_label(
    src_image_dir: str,
    src_mask_dir: str,
    json_path: str,
    target_base_dir: str,
    label_field: str,
) -> None:
    """
    按分类标签重新组织数据集。

    参数:
        src_image_dir  : 源图像目录
        src_mask_dir   : 源掩码目录
        json_path      : 分类标签 JSON 文件路径
        target_base_dir: 目标根目录
        label_field    : 用作分类标签的 JSON 字段名
    """
    # ---- 读取 JSON ----
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON 文件不存在: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"JSON 文件内容必须是数组(list): {json_path}")

    # 校验每条记录是否包含必需字段
    for i, rec in enumerate(records):
        if "filename" not in rec:
            raise KeyError(f"第 {i} 条记录缺少 'filename' 字段")
        if label_field not in rec:
            raise KeyError(f"第 {i} 条记录缺少 '{label_field}' 字段")

    # ---- 建立 mask 索引：stem（小写） -> mask 文件路径 ----
    mask_index: Dict[str, str] = {}
    for root, _, files in os.walk(src_mask_dir):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                stem, _ = os.path.splitext(fname)
                stem_lower = stem.lower()
                full_path = os.path.join(root, fname)
                if stem_lower in mask_index:
                    print(f"警告：mask stem 重复: {stem_lower}，"
                          f"旧={mask_index[stem_lower]}, 新={full_path}，保留旧的")
                else:
                    mask_index[stem_lower] = full_path

    print(f"mask 目录中共扫描到 {len(mask_index)} 个文件")

    # ---- 统计数据 ----
    label_counts: Dict[str, int] = defaultdict(int)
    skipped_no_mask = 0
    skipped_no_image = 0
    processed = 0

    # ---- 遍历每条记录 ----
    for rec in records:
        filename = rec["filename"]          # 可能含子目录路径
        label_value = rec[label_field]       # 分类标签值

        # 标签值转字符串，作为目录名
        label_dir = str(label_value)

        # 源图像路径
        src_img_path = os.path.join(src_image_dir, filename)
        if not os.path.isfile(src_img_path):
            print(f"警告：源图像不存在，跳过: {src_img_path}")
            skipped_no_image += 1
            continue

        # 根据文件名 stem 查找对应 mask
        leaf_filename = os.path.basename(filename)   # 纯文件名
        img_stem, img_ext = os.path.splitext(leaf_filename)
        img_stem_lower = img_stem.lower()

        mask_path = mask_index.get(img_stem_lower)
        if mask_path is None:
            print(f"警告：找不到对应掩码，stem={img_stem}，跳过: {leaf_filename}")
            skipped_no_mask += 1
            continue

        mask_leaf = os.path.basename(mask_path)

        # 目标路径
        dst_img_dir = os.path.join(target_base_dir, "images", label_dir)
        dst_mask_dir = os.path.join(target_base_dir, "masks", label_dir)
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_mask_dir, exist_ok=True)

        dst_img_path = os.path.join(dst_img_dir, leaf_filename)
        dst_mask_path = os.path.join(dst_mask_dir, mask_leaf)

        # 复制
        shutil.copy2(src_img_path, dst_img_path)
        shutil.copy2(mask_path, dst_mask_path)

        label_counts[label_dir] += 1
        processed += 1

    # ---- 打印统计信息 ----
    print(f"\n{'='*60}")
    print(f"处理完成，统计如下：")
    print(f"  分类标签字段: {label_field}")
    print(f"  成功处理条目: {processed}")
    print(f"  JSON 总记录数: {len(records)}")
    if skipped_no_image > 0:
        print(f"  因源图像不存在跳过: {skipped_no_image}")
    if skipped_no_mask > 0:
        print(f"  因找不到掩码跳过: {skipped_no_mask}")
    print(f"\n  各类别分布:")
    for label_val in sorted(label_counts.keys(), key=lambda x: (x.isdigit(), x)):
        print(f"    label={label_val}: {label_counts[label_val]} 对 (image+mask)")
    print(f"{'='*60}")


if __name__ == "__main__":
    reorganize_by_label(
        src_image_dir=SRC_IMAGE_DIR,
        src_mask_dir=SRC_MASK_DIR,
        json_path=JSON_PATH,
        target_base_dir=TARGET_BASE_DIR,
        label_field=LABEL_FIELD,
    )
