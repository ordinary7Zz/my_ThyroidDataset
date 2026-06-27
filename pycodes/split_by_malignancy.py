import json
import os
import shutil

def process_json(json_path, image_root_dir, output_root_dir=None):
    """
    json_path: 标签 json 路径，比如 DDTI_Classification_test_label.json
    image_root_dir: 原始图像所在目录（里面包含 10_1.jpg 这类文件）
    output_root_dir: 输出根目录（不填则默认用 image_root_dir）
    """
    if output_root_dir is None:
        output_root_dir = image_root_dir

    # 读 json
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_labels = len(data)
    print(f"Loaded {total_labels} records from {json_path}")

    copied_count = 0

    for item in data:
        filename = item["filename"]
        malignancy = item["malignancy"]  # 0 或 1

        src_path = os.path.join(image_root_dir, filename)
        target_dir = os.path.join(output_root_dir, str(malignancy))
        os.makedirs(target_dir, exist_ok=True)
        dst_path = os.path.join(target_dir, filename)

        if not os.path.exists(src_path):
            print(f"[WARN] Source image not found: {src_path}")
            continue

        shutil.copy2(src_path, dst_path)
        copied_count += 1

    # 复制完成后统计 0/1 目录中的文件总数
    count_0 = 0
    count_1 = 0
    dir_0 = os.path.join(output_root_dir, "0")
    dir_1 = os.path.join(output_root_dir, "1")

    if os.path.isdir(dir_0):
        count_0 = sum(
            1 for name in os.listdir(dir_0)
            if os.path.isfile(os.path.join(dir_0, name))
        )

    if os.path.isdir(dir_1):
        count_1 = sum(
            1 for name in os.listdir(dir_1)
            if os.path.isfile(os.path.join(dir_1, name))
        )

    total_in_dirs = count_0 + count_1

    print(f"Copied images: {copied_count}")
    print(f"Class 0 count in dir: {count_0}")
    print(f"Class 1 count in dir: {count_1}")
    print(f"Total in dirs (0+1): {total_in_dirs}")
    print(f"Total labels in json: {total_labels}")

    if total_in_dirs == total_labels:
        print("[CHECK] ✅ 目录中总文件数与标签总数一致。")
    else:
        print("[CHECK] ❌ 不一致，请检查缺失或重复：")
        print(f"       - 标签总数: {total_labels}")
        print(f"       - 目录总数: {total_in_dirs}")
        print(f"       - 实际成功复制: {copied_count}")

    print(f"Done for {json_path}.\n")

if __name__ == "__main__":
    # ====== 根据你自己的路径修改下面三项 ======

    # 1) json 文件列表（可以同时处理多个 json）
    json_paths = [
        "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all/DDTI_Classification_test_label.json",
        # 如果还有 train/val 的 json，在这里继续加
        # r"...\DDTI_Classification_train_label.json",
        # r"...\DDTI_Classification_val_label.json",
    ]

    # 2) 图像所在目录：里面应该有 10_1.jpg, 100_1.jpg 等文件
    image_root_dir = "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all/images_processed"

    # 3) 输出根目录（可以和 image_root_dir 一样，也可以单独指定一个新目录）
    output_root_dir = "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all_cls"

    # ====== 执行 ======
    for jp in json_paths:
        process_json(jp, image_root_dir, output_root_dir)