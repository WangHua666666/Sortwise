import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from ml.models.dataset import ImageDataset
from ml.models.transforms import ImageTransform
import timm

def train():
    """训练模型"""
    # 基本配置
    DATA_DIR = 'data/processed'
    MODEL_PATH = 'ml/models/checkpoints/best_model.pth'
    EPOCHS = 50
    BATCH_SIZE = 32
    
    # 创建保存目录
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler('train.log'), logging.StreamHandler()]
    )
    log = logging.getLogger()
    
    # 准备数据
    transform = ImageTransform()
    train_data = ImageDataset(DATA_DIR, transform.train_transform, 'train')
    val_data = ImageDataset(DATA_DIR, transform.val_transform, 'val')
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    
    # 创建模型
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=4)
    
    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 开始训练
    best_acc = 0
    log.info("开始训练...")
    
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = train_correct = train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f'训练 {epoch+1}/{EPOCHS}'):
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # 验证阶段
        model.eval()
        val_correct = val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'验证 {epoch+1}/{EPOCHS}'):
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # 计算准确率
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        # 记录结果
        log.info(f'Epoch {epoch+1}/{EPOCHS}:')
        log.info(f'训练 - 损失: {train_loss/len(train_loader):.4f}, 准确率: {train_acc:.2f}%')
        log.info(f'验证 - 准确率: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            log.info(f'保存最佳模型，准确率: {best_acc:.2f}%')

if __name__ == '__main__':
    train() 