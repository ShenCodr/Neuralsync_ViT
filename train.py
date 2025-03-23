import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter # 可视化工具
from torchvision import transforms# 图像预处理工具

from my_dataset import MyDataSet# 自定义数据集类，来自my_dataset.py
from vit_model import vit_base_patch16_224_in21k as create_model# ViT模型定义，来自vit_model.py
from utils import read_split_data, train_one_epoch, evaluate # 自定义工具函数,来自utils.py

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")# 设置运行设备（优先GPU）
    
    # 初始化TensorBoard记录器（用于可视化训练过程）
    tb_writer = SummaryWriter()

    ######################################################################################
    #                               数据准备阶段                                         #
    ######################################################################################
    # 读取并划分训练集/验证集数据路径和标签
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    # 定义训练集和验证集的数据预处理流程
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),# 随机裁剪缩放增强
            transforms.RandomHorizontalFlip(),# 水平翻转增强
            transforms.ToTensor(),            # 转换为张量
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])# 归一化
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),# 验证时调整尺寸
            transforms.CenterCrop(224), # 中心裁剪
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    # 初始化训练和验证数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])
    
    # 配置数据加载器参数
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])# 计算最佳并行加载进程数
    print('Using {} dataloader workers every process'.format(nw))
     # 创建训练和验证数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    ######################################################################################
    #                                 模型初始化阶段                                      #
    ######################################################################################
    # 创建ViT模型实例并转移到指定设备
    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

    # 如果提供预训练权重则加载
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重（分类头、预训练中不使用的部分）
        del_keys = ['head.weight', 'head.bias'] if model.has_logits else \
                   ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            if k in weights_dict:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    # 如果需要冻结部分层（例如冻结 backbone，只训练分类头）
    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
                
    ######################################################################################
    #                                优化器配置阶段                                       #
    ######################################################################################
    # 筛选需要更新的参数
    pg = [p for p in model.parameters() if p.requires_grad]
      # 使用SGD优化器（更适合迁移学习）
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
     # 余弦退火学习率调度器（结合lrf参数实现渐进式下降）
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine decay
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    ######################################################################################
    #                                  训练循环阶段                                       #
    ######################################################################################
    for epoch in range(args.epochs):
        # 训练单个epoch
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
         # 更新学习率
        scheduler.step()

        # 验证当前模型
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        # 记录训练指标到TensorBoard
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # 将模型保存到 checkpoints 文件夹中
        checkpoint_dir = r"D:\\VsCode\\ViT_model\\checkpoints"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        save_path = os.path.join(checkpoint_dir, "model-{}.pth".format(epoch))
        torch.save(model.state_dict(), save_path)
        print("Saved model at: ", save_path)


if __name__ == '__main__':
    ######################################################################################
    #                              命令行参数配置说明                                     #
    #            通过python train.py --参数名 参数值 的方式进行运行配置                  #
    ######################################################################################
    parser = argparse.ArgumentParser()
     # 分类任务类别数（默认couple和stranger两类）
    parser.add_argument('--num_classes', type=int, default=2)
    # 训练总轮次
    parser.add_argument('--epochs', type=int, default=20)
     # 批次大小
    parser.add_argument('--batch-size', type=int, default=8)
     # 初始学习率
    parser.add_argument('--lr', type=float, default=0.001)
     # 学习率最终衰减系数（lr_final = lr_initial * lrf）
    parser.add_argument('--lrf', type=float, default=0.01)
    # 数据集所在根目录
    parser.add_argument('--data_path', type=str,
                        default=r"D:\\EdgeDownLoad\\new_picture")
    parser.add_argument('--model-name', default='', help='create model name')
    # 预训练权重路径（默认为Google官方ImageNet21k预训练权重）
    parser.add_argument('--weights', type=str, 
                        default=r"D:\\VsCode\\ViT_model\\jx_vit_base_patch16_224_in21k-e5005f0a.pth",
                        help='initial weights path')
    # 是否冻结基础层（默认True，即仅训练分类头）
    parser.add_argument('--freeze-layers', type=bool, default=True)
    # 训练设备选择（默认优先使用GPU:0）
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
     # 解析参数并启动主流程
    opt = parser.parse_args()
    main(opt)
