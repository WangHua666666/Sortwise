import os
import sys
from pathlib import Path
from PIL import Image
import logging
import gc

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import time

from ml.models.efficientnet import WasteClassifier
from ml.config import Config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def print_gpu_memory():
    if torch.cuda.is_available():
        logging.info(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB allocated, "
                    f"{torch.cuda.memory_reserved()/1024**2:.1f}MB reserved")

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception as e:
        logging.error(f"损坏的图片文件 {path}: {str(e)}")
        return False

def pil_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except Exception as e:
        logging.error(f"无法加载图片 {path}: {str(e)}")
        # 创建一个1x1的黑色图片作为替代
        return Image.new('RGB', (1, 1), (0, 0, 0))

def train_model():
    try:
        # 1. 检查是否有GPU并清理内存
        device = Config.DEVICE
        clear_gpu_memory()
        print(f"Using device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print_gpu_memory()

        # 2. 数据预处理
        transform = transforms.Compose([
            transforms.Resize((Config.DATA_CONFIG['image_size'], Config.DATA_CONFIG['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.DATA_CONFIG['mean'],
                              std=Config.DATA_CONFIG['std'])
        ])

        # 3. 加载数据
        train_dir = Config.DATA_CONFIG['train_dir']
        if not os.path.exists(train_dir):
            print(f"错误: 找不到训练数据目录 '{train_dir}'")
            print("请确保数据目录结构正确")
            sys.exit(1)

        print("正在检查训练数据完整性...")
        valid_files = []
        for root, _, files in os.walk(train_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    file_path = os.path.join(root, file)
                    if is_valid_image(file_path):
                        valid_files.append(file_path)
                    else:
                        logging.warning(f"跳过损坏的文件: {file_path}")

        print(f"找到 {len(valid_files)} 个有效的训练图片")
        
        print("正在加载训练数据...")
        train_dataset = datasets.ImageFolder(
            train_dir,
            transform=transform,
            loader=pil_loader
        )
        
        print(f"找到 {len(train_dataset)} 张训练图片")
        print("类别信息:")
        for class_idx, class_name in enumerate(train_dataset.classes):
            class_count = len([x for x, y in train_dataset.samples if y == class_idx])
            print(f"  - {class_idx}: {class_name}: {class_count} 张图片")
        
        print(f"\n总类别数: {len(train_dataset.classes)}")
        print(f"类别到索引的映射:")
        print(train_dataset.class_to_idx)
        
        # 更新配置中的类别数
        Config.MODEL_CONFIG['num_classes'] = len(train_dataset.classes)
        
        # 4. 创建数据加载器
        # 减小批量大小以避免内存问题
        batch_size = min(16, Config.adjust_batch_size())
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=min(2, Config.TRAIN_CONFIG['num_workers']),  # 减少worker数量
            pin_memory=Config.TRAIN_CONFIG['pin_memory']
        )

        # 5. 创建模型
        print("\n正在初始化模型...")
        model = WasteClassifier(
            num_classes=Config.MODEL_CONFIG['num_classes'],
            model_name=Config.MODEL_CONFIG['model_name'],
            pretrained=Config.MODEL_CONFIG['pretrained']
        )
        model = model.to(device)
        
        # 6. 设置优化器和损失函数
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=Config.TRAIN_CONFIG['learning_rate'],
            weight_decay=Config.TRAIN_CONFIG['weight_decay']
        )
        criterion = nn.CrossEntropyLoss()

        # 设置梯度裁剪
        max_grad_norm = 1.0

        # 7. 训练循环
        num_epochs = Config.TRAIN_CONFIG['num_epochs']
        best_acc = 0.0
        train_start_time = time.time()
        
        print("\n开始训练...")
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # 使用tqdm显示进度条
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for batch_idx, (inputs, labels) in enumerate(pbar):
                try:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    
                    # 应用梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'loss': f'{running_loss/(batch_idx+1):.4f}',
                        'acc': f'{100.*correct/total:.2f}%'
                    })
                    
                    # 定期清理GPU内存
                    if batch_idx % 10 == 0:
                        clear_gpu_memory()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logging.error("GPU内存不足，尝试清理内存并继续...")
                        clear_gpu_memory()
                        continue
                    else:
                        raise e
            
            # 计算epoch统计信息
            epoch_loss = running_loss/len(train_loader)
            epoch_acc = 100.*correct/total
            
            print(f'\nEpoch {epoch+1} 统计:')
            print(f'Loss: {epoch_loss:.4f}')
            print(f'Accuracy: {epoch_acc:.2f}%')
            print_gpu_memory()
            
            # 保存最佳模型
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                print(f'发现更好的模型! 准确率: {best_acc:.2f}%')
                os.makedirs(str(Config.MODEL_WEIGHTS_DIR), exist_ok=True)
                torch.save(model.state_dict(), 
                         str(Config.get_best_model_path()))
            
            # 每个epoch结束后清理内存
            clear_gpu_memory()

    except KeyboardInterrupt:
        print("\n训练被用户中断")
        print("保存当前模型...")
        os.makedirs(str(Config.MODEL_WEIGHTS_DIR), exist_ok=True)
        torch.save(model.state_dict(), 
                  str(Config.get_interrupted_model_path()))
    
    except Exception as e:
        print(f"\n训练过程出错: {str(e)}")
        print_gpu_memory()
        raise

    finally:
        clear_gpu_memory()

if __name__ == '__main__':
    train_model() 