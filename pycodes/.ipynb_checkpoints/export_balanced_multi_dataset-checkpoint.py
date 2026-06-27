from __future__ import annotations

"""从多个数据集的测试标签 JSON 中做二分类均衡抽样，并导出统一的新测试集。

脚本用途：
1. 读取多个数据集各自的测试标签 JSON
2. 按 TARGET_LABEL 过滤掉标签为 -1 的样本
3. 从每个数据集内分别随机抽取指定数量的样本，并保证该数据集内 0/1 数量一致
4. 合并所有抽样结果，再检查总结果的 0/1 数量也一致
5. 把抽中的图像统一复制到新目录，重命名为 001、002、003...
6. 生成新的 JSON 标签文件，格式保持和原始标签文件一致
7. 生成新的 CSV 文件，只保留两列：filename 和目标标签值

这个脚本不使用命令行参数，所有配置都直接写在下面这些常量里。
修改完常量后，直接运行脚本即可。
"""

import csv
import json
import random
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# 每个元素对应一个数据集的测试标签 JSON 路径。
# JSON 内容格式需要与 test_labels.json 相同，至少包含 filename 和目标标签字段。
JSON_PATHS = [
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/TN3K_test_label.json",
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/ThyroidXL_test_label.json",
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/TN5K_test_label.json",
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all/DDTI_Classification_test_label.json",
]

# 每个元素表示从对应 JSON 中要抽取多少条数据。
# 每个数量都必须是偶数，这样才能保证每个数据集内部 0/1 完全均匀。
SAMPLE_COUNTS = [
    80, 250, 140, 30,
]

# 每个元素对应一个数据集的图像根目录。
# 程序会用 dataset_root / filename 拼出原始图像文件的完整路径。
DATASET_ROOTS = [
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/images",
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/images",
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/images",
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all/images_processed",
]

# 要做均衡抽样的标签名称。
# 这个标签必须是二分类任务，且标签值只能是 0、1、-1。
TARGET_LABEL = "malignancy"

# 固定随机种子后，每次运行都能得到相同的抽样结果，便于复现。
RANDOM_SEED = 42

# 导出目录和输出文件路径。
# 图像会被复制到 OUTPUT_DIR 中，JSON 和 CSV 保存到上一级 merged_test_set 目录下。
OUTPUT_DIR = Path("/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/merged_test_set/images")
OUTPUT_JSON_PATH = Path("/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/merged_test_set/merged_labels.json")
OUTPUT_CSV_PATH = Path("/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/merged_test_set/merged_labels.csv")


def _validate_config() -> None:
    """检查顶部常量配置是否合法。

    这里主要检查三件事：
    1. 三个配置数组不能为空且长度一致
    2. 每个数据集的抽样数量必须大于 0
    3. 每个数据集的抽样数量、以及总抽样数量，都必须是偶数

    这样才能保证每个数据集内部以及最终合并后的 0/1 数量可以做到完全均匀。
    """
    if not JSON_PATHS:
        raise ValueError("JSON_PATHS 不能为空")
    if not (len(JSON_PATHS) == len(SAMPLE_COUNTS) == len(DATASET_ROOTS)):
        raise ValueError("JSON_PATHS、SAMPLE_COUNTS、DATASET_ROOTS 的长度必须一致")
    if any(count <= 0 for count in SAMPLE_COUNTS):
        raise ValueError("SAMPLE_COUNTS 中的数量必须都大于 0")
    if any(count % 2 != 0 for count in SAMPLE_COUNTS):
        raise ValueError("每个数据集的抽样数量必须是偶数，才能保证每个数据集内 0/1 完全均匀")
    if sum(SAMPLE_COUNTS) % 2 != 0:
        raise ValueError("总抽样数量必须是偶数，才能保证最终 0/1 完全均匀")


def _load_json(path: Path) -> list[dict]:
    """读取一个标签 JSON 文件，并确保最外层结构是列表。"""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"JSON 结构错误，必须是列表: {path}")
    return data


def _split_by_label(rows: list[dict], label_name: str) -> tuple[list[dict], list[dict]]:
    """按指定标签把记录拆成 0 类和 1 类，并过滤掉标签为 -1 的记录。"""
    zero_rows: list[dict] = []
    one_rows: list[dict] = []

    for row in rows:
        if label_name not in row:
            raise KeyError(f"标签不存在: {label_name}")
        value = row[label_name]
        if value == -1:
            continue
        if value == 0:
            zero_rows.append(row)
        elif value == 1:
            one_rows.append(row)
        else:
            raise ValueError(f"标签 {label_name} 只能是 0、1、-1，实际得到: {value}")

    return zero_rows, one_rows


def _sample_balanced_rows(
    rows: list[dict],
    label_name: str,
    sample_count: int,
    rng: random.Random,
) -> list[dict]:
    """从单个数据集里按指定标签做均衡随机抽样。

    例如 sample_count=100 时，会从标签 0 中随机抽 50 条，
    从标签 1 中随机抽 50 条，然后打乱顺序后返回。
    """
    zero_rows, one_rows = _split_by_label(rows, label_name)
    half = sample_count // 2

    if len(zero_rows) < half or len(one_rows) < half:
        raise ValueError(
            f"无法从当前数据集中按标签 {label_name} 均匀抽取 {sample_count} 条。"
            f"可用数量: 0类={len(zero_rows)}, 1类={len(one_rows)}"
        )

    sampled_zero = rng.sample(zero_rows, half)
    sampled_one = rng.sample(one_rows, half)
    sampled_rows = sampled_zero + sampled_one
    rng.shuffle(sampled_rows)
    return sampled_rows


def _build_source_path(dataset_root: Path, row: dict) -> Path:
    """根据数据集根目录和 JSON 记录中的 filename 拼出原始图像路径。"""
    if "filename" not in row:
        raise KeyError("JSON 记录缺少 filename 字段")
    return dataset_root / Path(row["filename"])


def export_balanced_multi_dataset() -> None:
    """执行完整导出流程。

    执行步骤如下：
    1. 检查配置是否正确
    2. 逐个读取各数据集标签文件
    3. 在每个数据集内按目标标签均衡抽样
    4. 合并全部抽样结果，并再次检查总标签分布是否均衡
    5. 复制原始图像到输出目录，统一重命名为 001、002、003...
    6. 生成新的 JSON 和 CSV 标签文件
    """
    _validate_config()
    rng = random.Random(RANDOM_SEED)

    selected_rows: list[dict] = []
    source_paths: list[Path] = []

    for json_path, sample_count, dataset_root in zip(JSON_PATHS, SAMPLE_COUNTS, DATASET_ROOTS):
        json_path = Path(json_path).resolve()
        dataset_root = Path(dataset_root).resolve()

        rows = _load_json(json_path)
        sampled_rows = _sample_balanced_rows(rows, TARGET_LABEL, sample_count, rng)

        for row in sampled_rows:
            source_path = _build_source_path(dataset_root, row)
            if not source_path.is_file():
                raise FileNotFoundError(f"找不到图像文件: {source_path}")
            selected_rows.append(dict(row))
            source_paths.append(source_path)

    zero_count = sum(1 for row in selected_rows if row[TARGET_LABEL] == 0)
    one_count = sum(1 for row in selected_rows if row[TARGET_LABEL] == 1)
    if zero_count != one_count:
        raise ValueError(
            f"最终导出结果不均匀: {TARGET_LABEL}=0 有 {zero_count} 条, {TARGET_LABEL}=1 有 {one_count} 条"
        )

    paired_rows = list(zip(selected_rows, source_paths))
    rng.shuffle(paired_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    width = max(3, len(str(len(paired_rows))))

    exported_rows: list[dict] = []
    for index, (row, source_path) in enumerate(paired_rows, start=1):
        suffix = source_path.suffix.lower() or ".jpg"
        new_filename = f"{index:0{width}d}{suffix}"
        target_path = OUTPUT_DIR / new_filename

        shutil.copy2(source_path, target_path)

        new_row = dict(row)
        new_row["filename"] = new_filename
        exported_rows.append(new_row)

    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(exported_rows, f, ensure_ascii=False, indent=2)
        f.write("\n")

    with OUTPUT_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", TARGET_LABEL])
        for row in exported_rows:
            writer.writerow([row["filename"], row[TARGET_LABEL]])

    print(f"标签名称: {TARGET_LABEL}")
    print(f"导出图像数量: {len(exported_rows)}")
    print(f"标签 0 数量: {zero_count}")
    print(f"标签 1 数量: {one_count}")
    print(f"图像输出目录: {OUTPUT_DIR}")
    print(f"JSON 输出文件: {OUTPUT_JSON_PATH}")
    print(f"CSV 输出文件: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    export_balanced_multi_dataset()
