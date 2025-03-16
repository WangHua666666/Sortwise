import os
import json
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    """图像分类数据集"""
    
    def __init__(self, data_dir, transform=None, split='train'):
        """初始化数据集
        
        Args:
            data_dir: 数据目录
            transform: 图像转换
            split: 数据集划分('train' or 'val')
        """
        self.transform = transform
        
        # 加载标注文件
        anno_path = os.path.join(data_dir, f'{split}_annotations.json')
        with open(anno_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # 过滤有效图片
        self.samples = [
            (os.path.join(data_dir, item['image_path']), item['label'])
            for item in annotations
            if os.path.exists(os.path.join(data_dir, item['image_path']))
        ]
        
        print(f"加载{split}数据集: {len(self.samples)}张图片")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, index):
        """获取一个样本
        
        Args:
            index: 样本索引
            
        Returns:
            tuple: (图像张量, 标签)
        """
        img_path, label = self.samples[index]
        
        try:
            # 加载和转换图片
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except:
            # 发生错误时返回第一个样本
            return self[0] 