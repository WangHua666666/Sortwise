import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from ml.models.efficientnet import get_model
from ml.config import Config

# 图像预处理转换
transform = transforms.Compose([
    # 调整图像大小为模型配置中指定的尺寸
    transforms.Resize((Config.DATA_CONFIG['image_size'], Config.DATA_CONFIG['image_size'])),
    # 转换为张量
    transforms.ToTensor(),
    # 标准化：使用ImageNet数据集的均值和标准差
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# 类别映射字典
categories = {
    0: '可回收物',  # 如：纸张、塑料、金属、玻璃
    1: '有害垃圾',  # 如：电池、药品、油漆
    2: '厨余垃圾',  # 如：食物残渣、果皮
    3: '其他垃圾'   # 如：卫生纸、尘土
}

def load_model():
    """
    加载预训练的深度学习模型
    
    Returns:
        torch.nn.Module: 加载好的模型实例
        
    注意：
        - 使用配置类中指定的模型路径
        - 使用配置类中指定的设备进行推理
        - 模型加载后自动设置为评估模式
    """
    # 获取模型实例
    model = get_model(
        num_classes=len(Config.CLASSES),
        weights_path=str(Config.get_best_model_path()),
        device=Config.DEVICE
    )
    # 设置为评估模式
    model.eval()
    return model

def get_prediction(image_path):
    """
    对输入图片进行废弃物分类预测
    
    Args:
        image_path (str): 图片文件路径
        
    Returns:
        dict: 预测结果，包含以下字段：
            - category_id: 类别ID
            - category_name: 类别名称
            - confidence: 置信度（百分比）
            
    Raises:
        Exception: 当图片加载或预测过程出错时抛出
    """
    try:
        # 加载并预处理图片
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
        
        # 将图像数据移动到指定设备
        image_tensor = image_tensor.to(Config.DEVICE)

        # 加载模型
        model = load_model()

        # 进行预测
        with torch.no_grad():  # 不计算梯度
            # 获取模型输出
            outputs = model(image_tensor)
            # 获取预测类别
            _, predicted = torch.max(outputs, 1)
            # 计算置信度
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence = probabilities.max().item()

        # 获取预测结果
        category_id = predicted.item()
        category_name = Config.CLASSES[category_id]

        return {
            'category_id': category_id,
            'category_name': category_name,
            'confidence': round(confidence * 100, 2)  # 转换为百分比
        }

    except Exception as e:
        raise Exception(f"预测过程出错: {str(e)}") 