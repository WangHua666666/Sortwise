import os
import json
import shutil
from pathlib import Path
import random
from tqdm import tqdm

def prepare_dataset(
    source_dir: str,
    target_dir: str,
    mapping_file: str,
    train_ratio: float = 0.8,
    seed: int = 42
):
    """
    准备数据集，包括：
    1. 将图片按类别整理到processed目录
    2. 生成训练集和验证集的标注文件
    3. 生成数据集统计信息
    """
    # 设置随机种子
    random.seed(seed)
    
    # 创建必要的目录
    processed_dir = Path(target_dir)
    images_dir = processed_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载类别映射
    with open(mapping_file, 'r', encoding='utf-8') as f:
        category_mapping = json.load(f)
    
    # 创建反向映射：从具体类别到四大类
    reverse_mapping = {}
    for main_category, sub_categories in category_mapping.items():
        for sub_category in sub_categories:
            reverse_mapping[sub_category.lower()] = main_category
    
    # 用于存储标注信息
    all_annotations = []
    
    # 遍历源数据目录
    source_path = Path(source_dir)
    for sub_category in tqdm(os.listdir(source_dir), desc="处理数据集"):
        if not (source_path / sub_category).is_dir():
            continue
            
        # 获取主类别
        sub_category_lower = sub_category.lower()
        if sub_category_lower not in reverse_mapping:
            print(f"警告：找不到类别映射 {sub_category}，跳过")
            continue
            
        main_category = reverse_mapping[sub_category_lower]
        main_category_id = list(category_mapping.keys()).index(main_category)
        
        # 获取该类别下的所有图片
        category_dir = source_path / sub_category
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(category_dir.glob(f"*{ext}")))
            image_files.extend(list(category_dir.glob(f"*{ext.upper()}")))
        
        # 为每张图片创建标注
        for img_path in image_files:
            try:
                # 生成新的文件名，确保唯一性
                new_filename = f"{sub_category}_{img_path.name}"
                new_image_path = images_dir / new_filename
                
                # 复制图片文件
                shutil.copy2(img_path, new_image_path)
                
                # 创建标注信息
                annotation = {
                    "image_path": f"images/{new_filename}",
                    "label": main_category_id,
                    "category": main_category,
                    "sub_category": sub_category
                }
                all_annotations.append(annotation)
            except Exception as e:
                print(f"处理图片时出错 {img_path}: {str(e)}")
    
    # 随机打乱数据
    random.shuffle(all_annotations)
    
    # 划分训练集和验证集
    split_idx = int(len(all_annotations) * train_ratio)
    train_annotations = all_annotations[:split_idx]
    val_annotations = all_annotations[split_idx:]
    
    # 保存标注文件
    with open(processed_dir / 'train_annotations.json', 'w', encoding='utf-8') as f:
        json.dump(train_annotations, f, ensure_ascii=False, indent=4)
        
    with open(processed_dir / 'val_annotations.json', 'w', encoding='utf-8') as f:
        json.dump(val_annotations, f, ensure_ascii=False, indent=4)
    
    # 打印数据集统计信息
    print("\n数据集准备完成！")
    print(f"训练集数量：{len(train_annotations)}")
    print(f"验证集数量：{len(val_annotations)}")
    
    # 打印每个类别的数量统计
    print("\n类别分布：")
    category_counts = {category: {'train': 0, 'val': 0} for category in category_mapping.keys()}
    
    for annotation in train_annotations:
        category = annotation['category']
        category_counts[category]['train'] += 1
        
    for annotation in val_annotations:
        category = annotation['category']
        category_counts[category]['val'] += 1
    
    for category, counts in category_counts.items():
        total = counts['train'] + counts['val']
        print(f"\n{category}:")
        print(f"  训练集: {counts['train']} 张图片")
        print(f"  验证集: {counts['val']} 张图片")
        print(f"  总计: {total} 张图片")

if __name__ == '__main__':
    prepare_dataset(
        source_dir='data/train',
        target_dir='data/processed',
        mapping_file='data/processed/category_mapping.json'
    ) 