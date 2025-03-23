from PIL import Image
import torch
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    """自定义PyTorch数据集类，用于将图片数据和标签组织成PyTorch可直接使用的Dataset格式
    
    功能:
    1. 管理图像文件路径和对应标签
    2. 动态加载图像数据
    3. 提供数据预处理接口
    4. 实现数据批量生成的适配方法
    """

    def __init__(self, images_path: list, images_class: list, transform=None):
        """
        初始化数据集实例
        
        参数说明：
        images_path  : list, 包含所有图像文件路径的列表（例如 ["data/001.jpg", "data/002.jpg"...]）
        images_class : list, 与图片路径一一对应的类别标签列表（例如 [0, 1, 0,...]）
        transform    : 可选，数据增强/预处理操作（如 torchvision.transforms 的组合操作）
        """
        self.images_path = images_path  # 图像文件路径存储
        self.images_class = images_class  # 类别标签存储
        self.transform = transform  # 预处理操作初始化

    def __len__(self):
        """返回数据集的总样本数，供DataLoader获取数据量"""
        return len(self.images_path)

    def __getitem__(self, item):
        """核心方法：根据索引获取单个样本（图像+标签），含数据加载与预处理
        
        工作流程：
        1. 按索引加载图片 -> 2. 检查格式 -> 3. 提取标签 -> 4. 应用数据增强
        """
        # 步骤1：使用PIL库加载指定路径的图片（支持JPG/PNG等常见格式）
        img = Image.open(self.images_path[item])
        
        # 防御性编程：强制校验图像格式
        # - RGB模式: 彩色图像（3通道）
        # - L模式  : 灰度图像（1通道）
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        
        # 从预存列表中获取对应标签（类型根据具体任务而定，可能是int/float等）
        label = self.images_class[item]

        # 步骤2：应用数据预处理/增强流程（如果有配置）
        if self.transform is not None:
            img = self.transform(img)  # 通常包括：归一化、随机裁剪、颜色变换等

        return img, label  # 返回处理后的张量图像和对应标签

    @staticmethod
    def collate_fn(batch):
        """自定义批次数据组合方法，适配DataLoader的collate_fn参数
        
        特殊处理需求：
        - 当图像原始尺寸不一致时，需要统一尺寸（但当前实现假设预处理已完成尺寸统一）
        - 将Python列表的多个(image, label)元组转换为批量张量
        """
        # 解压批次数据：把多个(image, label)元组拆解成images列表和labels列表
        # 例如：输入 [(img1,1), (img2,0)] -> images=[img1,img2], labels=[1,0]
        images, labels = tuple(zip(*batch))

        # 关键操作：将多个图像张量堆叠成批量维度
        # 假设所有图像已经预处理为相同尺寸，则直接堆叠成 [batch_size, channels, H, W]
        images = torch.stack(images, dim=0)
        
        # 将标签列表转换为张量（支持数字/类别的自动转换）
        labels = torch.as_tensor(labels)  # 输出形状：[batch_size, ]
        
        return images, labels  # 返回标准的批量张量格式
