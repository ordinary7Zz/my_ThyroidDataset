"""
================================================================================
脚本名称：create_all_dataset_cls.py
功能说明：将多个子数据集的图像和标注掩码（mask）汇总到一个目录，并可合并 JSON 标签文件。

核心逻辑：
    1. 从多个源 image_dirs 和 mask_dirs 中，按文件名（不含扩展名）一一匹配图像与掩码，
       复制到目标 dst_image_dir 和 dst_mask_dir。
    2. 如果提供了 json_paths 列表，将多个 JSON 文件（每个是 list 结构）合并成一个
       大的 JSON 数组，写入 out_json_path。
    3. 最后检查目标目录中图像数量、掩码数量、JSON 条目数是否三者一致。

适用场景：
    - 你有多个来源的甲状腺超声数据集（如 TN3K、ThyroidXL、TN5K、Cine-Clip 等），
      每个数据集有自己的 images、masks 目录和 label.json 文件，需要合并成一个
      统一的大数据集，用于分类任务的训练。

使用方法：
    1. 修改 main 块中的四个配置变量：
       - image_dirs: 源图像目录列表，每个目录下包含 .jpg/.jpeg/.png 图像文件。
       - mask_dirs: 源掩码目录列表，与 image_dirs 一一对应。
       - dst_image_dir: 汇总后的目标图像目录。
       - dst_mask_dir: 汇总后的目标掩码目录。
       - json_paths: 要合并的 JSON 标签文件列表（可选）。
       - out_json_path: 合并后的 JSON 文件输出路径（json_paths 非空时必填）。

    2. 确保 image_dirs 和 mask_dirs 长度相等，且按相同索引一一对应。
       例如 image_dirs[0] 中的图像，会在 mask_dirs[0] 中寻找同名掩码。

    3. 如果不需要合并 JSON，将 json_paths 设为空列表 [] 即可，此时不会执行
       JSON 合并步骤，也不会进行图像数量与 JSON 条目数的校验。

    4. 运行脚本：
           python create_all_dataset_cls.py

    5. 运行完成后会打印目标目录的文件数量，以及合并 JSON 的条目数。
       如果 images、masks、JSON 数量不一致，会抛出 ValueError。

手动调用函数示例：
    from create_all_dataset_cls import copy_images_and_merge_json

    copy_images_and_merge_json(
        image_dirs=["/data/A/images", "/data/B/images"],
        mask_dirs=["/data/A/masks", "/data/B/masks"],
        dst_image_dir="/data/merged/images",
        dst_mask_dir="/data/merged/masks",
        json_paths=["/data/A/label.json", "/data/B/label.json"],
        out_json_path="/data/merged/label.json",
    )

注意事项：
    - 图像和掩码的匹配依据是文件名（不含扩展名），例如 a.jpg 匹配 a.png。
    - 大小写不敏感。
    - 若同一 stem 下有多个候选掩码文件，只取第一个。
    - 若某图像找不到对应掩码，会打印警告并跳过该图像。
    - 复制使用 shutil.copy2，会保留文件的元数据（如修改时间）。
================================================================================
"""

import os
import shutil
import json
from typing import List, Optional, Iterable


def copy_images_and_merge_json(
    image_dirs: List[str],
    mask_dirs: List[str],
    dst_image_dir: str,
    dst_mask_dir: str,
    json_paths: Optional[List[str]] = None,
    out_json_path: Optional[str] = None,
) -> None:
    """
    1. 将 image_dirs 和 mask_dirs 中各个目录下的图像文件复制到目标目录（保持文件名不变）。
    2. 若 json_paths 不为空，将这些 JSON 文件中数组里的元素拼接后写到 out_json_path。
    """

    if len(image_dirs) != len(mask_dirs):
        raise ValueError("image_dirs 和 mask_dirs 的长度必须相同")

    # 创建目标目录（若不存在）
    os.makedirs(dst_image_dir, exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)

    def iter_image_files(directory: str) -> Iterable[str]:
        for root, _, files in os.walk(directory):
            for fname in files:
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    yield os.path.join(root, fname)

    # 复制图像和 mask
    for img_dir, msk_dir in zip(image_dirs, mask_dirs):
        if not os.path.isdir(img_dir):
            raise NotADirectoryError(f"image 目录不存在或不是目录: {img_dir}")
        if not os.path.isdir(msk_dir):
            raise NotADirectoryError(f"mask 目录不存在或不是目录: {msk_dir}")

        # 先为当前 mask 目录建立一个“无扩展名 -> 文件路径”索引
        mask_index = {}
        for m in iter_image_files(msk_dir):
            m_name = os.path.basename(m)
            stem, _ = os.path.splitext(m_name)
            stem = stem.lower()
            # 理论上同一个 stem 只会有一个文件，这里用 list 以防万一
            mask_index.setdefault(stem, []).append(m)

        for img in iter_image_files(img_dir):
            img_name = os.path.basename(img)
            img_stem, _ = os.path.splitext(img_name)
            img_stem_key = img_stem.lower()

            candidates = mask_index.get(img_stem_key)
            if not candidates:
                print(
                    f"警告：在 mask 目录中找不到同名（不含扩展名）文件，"
                    f"image={img_name}, mask_dir={msk_dir}"
                )
                continue

            # 若有多个候选，简单取第一个
            msk = candidates[0]
            msk_name = os.path.basename(msk)

            shutil.copy2(img, os.path.join(dst_image_dir, img_name))
            shutil.copy2(msk, os.path.join(dst_mask_dir, msk_name))

    # 复制完成后统计目标目录中的图像数量
    dst_image_count = sum(1 for _ in iter_image_files(dst_image_dir))
    dst_mask_count = sum(1 for _ in iter_image_files(dst_mask_dir))

    json_count: Optional[int] = None

    # 处理 JSON 合并
    if json_paths:
        if out_json_path is None:
            raise ValueError("json_paths 非空时必须提供 out_json_path")

        merged: List[dict] = []

        for jp in json_paths:
            if not os.path.exists(jp):
                raise FileNotFoundError(f"JSON 文件不存在: {jp}")

            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError(f"JSON 文件内容必须是数组(list)，当前文件: {jp}")

            merged.extend(data)

        # 写出新的 JSON 文件（数组）
        out_dir = os.path.dirname(out_json_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        json_count = len(merged)

    # 打印两个目标目录和 JSON（若有）的数量信息
    print(f"目标 images 目录: {dst_image_dir}, 文件数量: {dst_image_count}")
    print(f"目标 masks 目录:  {dst_mask_dir}, 文件数量: {dst_mask_count}")
    if json_count is not None:
        print(f"合并后 JSON 文件: {out_json_path}, 条目数量: {json_count}")
        
    # 最终数量一致性检查
    if dst_image_count != dst_mask_count:
        raise ValueError(
            f"目标目录中 images 与 masks 数量不一致："
            f"images={dst_image_count}, masks={dst_mask_count}"
        )

    if json_count is not None and dst_image_count != json_count:
        raise ValueError(
            f"目标目录中图像数量与 JSON 条目数量不一致："
            f"images={dst_image_count}, json_items={json_count}"
        )


if __name__ == "__main__":
    # ======= 1. 在这里配置你的源数据“目录数组” =======
    # 现在要求：数组元素是目录路径，每个目录下面包含很多图像文件。
    # 例如有多个子数据集，每个都有 images 和 masks 两个目录：
    image_dirs: List[str] = [
        "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/train/images",
        "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/train/images",
        "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/train/images",
        "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/train/images",
    ]

    mask_dirs: List[str] = [
        "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/train/masks",
        "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/train/masks",
        "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/train/masks",
        "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/train/masks",
    ]

    # ======= 2. 配置要复制到的目标路径 =======
    dst_image_dir = "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/all_datasets_cls/images"
    dst_mask_dir = "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/all_datasets_cls/masks"

    # ======= 3. 配置 JSON 合并相关路径（可以为空） =======
    # 如果你有多个 json 文件要合并，像 tn3k_test_label.json 这种
    json_paths: List[str] = [
        "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/train/TN3K_train_label.json",
        "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/train/ThyroidXL_train_label.json",
        "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/train/TN5K_train_label.json",
        "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/train/Cine-Clip_train_label.json",
    ]

    # 如果不需要 json 合并，可以把 json_paths 设为 [] ，或者直接注释掉下两行：
    out_json_path = "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/all_datasets_cls/all_datasets_label.json"

    # ======= 4. 调用函数（真正执行复制和合并） =======
    copy_images_and_merge_json(
        image_dirs=image_dirs,
        mask_dirs=mask_dirs,
        dst_image_dir=dst_image_dir,
        dst_mask_dir=dst_mask_dir,
        json_paths=json_paths if json_paths else None,
        out_json_path=out_json_path if json_paths else None,
    )

    print("数据复制与 JSON 合并完成。")