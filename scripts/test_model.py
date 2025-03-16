import os
import sys
import json
from PIL import Image
import matplotlib.pyplot as plt

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)

from ml.models.inference import GarbagePredictor

def test_single_image(image_path):
    """测试单张图片"""
    # 创建预测器
    predictor = GarbagePredictor()
    
    # 进行预测
    result = predictor.predict(image_path)
    
    # 显示结果
    if result['success']:
        # 显示图片
        img = Image.open(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"预测类别: {result['category']}\n置信度: {result['confidence']:.2%}")
        plt.show()
    else:
        print(f"预测失败: {result['error']}")

def test_validation_set(data_dir='data/processed', num_samples=5):
    """测试验证集中的样本"""
    # 加载验证集标注
    with open(os.path.join(data_dir, 'val_annotations.json'), 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # 创建预测器
    predictor = GarbagePredictor()
    
    # 随机选择样本进行测试
    import random
    samples = random.sample(annotations, num_samples)
    
    correct = 0
    for sample in samples:
        image_path = os.path.join(data_dir, sample['image_path'])
        true_label = sample['category']
        
        # 预测
        result = predictor.predict(image_path)
        
        if result['success']:
            pred_label = result['category']
            confidence = result['confidence']
            
            # 统计正确率
            if pred_label == true_label:
                correct += 1
            
            # 显示结果
            img = Image.open(image_path)
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"预测类别: {pred_label} (置信度: {confidence:.2%})\n真实类别: {true_label}")
            plt.show()
        else:
            print(f"预测失败: {result['error']}")
    
    # 打印准确率
    accuracy = correct / num_samples
    print(f"\n测试样本数: {num_samples}")
    print(f"准确率: {accuracy:.2%}")

if __name__ == '__main__':
    # 测试单张图片
    # test_single_image('path_to_test_image.jpg')
    
    # 测试验证集样本
    test_validation_set(num_samples=5) 