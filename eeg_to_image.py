import os
import sys
import time
import glob
import re
from pathlib import Path

print("开始导入库...")

try:
    import numpy as np
    print("numpy导入成功")
    import matplotlib.pyplot as plt
    print("matplotlib导入成功")
    import mne
    print("mne导入成功")
    from scipy.signal import welch
    print("scipy.signal导入成功")
    from pathlib import Path
    print("pathlib导入成功")
    from sklearn.preprocessing import MinMaxScaler
    print("sklearn导入成功")
except Exception as e:
    print(f"导入错误: {e}")
    sys.exit(1)
    
print("所有库导入成功!\n")

def load_eeg_data(eeg_file):
    """
    加载脑电数据，支持.set/.fdt、.vhdr或.fat格式
    
    参数:
        eeg_file: 脑电数据文件路径
        
    返回:
        raw: MNE Raw对象
    """
    print(f"正在加载脑电数据: {eeg_file}")
    try:
        # 检查文件格式
        file_ext = os.path.splitext(eeg_file)[1].lower()
        
        if file_ext == '.set':
            # EEGLAB格式文件加载
            raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
        elif file_ext == '.vhdr':
            # BrainVision格式文件加载
            raw = mne.io.read_raw_brainvision(eeg_file, preload=True)
        elif file_ext == '.fat':
            # 尝试作为EEGLAB数据加载.fat文件
            # 通常.fat与.set是一对文件，.set包含头信息，.fat包含数据
            set_file = eeg_file.replace('.fat', '.set')
            if os.path.exists(set_file):
                raw = mne.io.read_raw_eeglab(set_file, preload=True)
            else:
                # 如果没有对应的.set文件，直接尝试加载.fat
                try:
                    raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
                except:
                    raise ValueError(f"无法加载.fat格式文件: {eeg_file}")
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")
            
        print(f"成功加载数据: {len(raw.times)}秒, {len(raw.ch_names)}通道")
        return raw
    except Exception as e:
        print(f"加载脑电数据失败: {e}")
        raise

def create_eeg_images(raw1, raw2, output_dir, subject_pair_name, n_segments=20):
    """
    创建连通性矩阵图像
    
    参数:
        raw1: 第一个被试的脑电数据
        raw2: 第二个被试的脑电数据
        output_dir: 输出目录
        subject_pair_name: 被试对名称
        n_segments: 将数据分成的段数
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 仅创建连接性矩阵图像
    create_connectivity_images(raw1, raw2, output_dir, subject_pair_name, n_segments)

def create_connectivity_images(raw1, raw2, output_dir, subject_pair_name, n_segments):
    """创建连接性矩阵图像"""
    print("正在创建连接性矩阵图像...")
    
    # 获取数据
    data1 = raw1.get_data()
    data2 = raw2.get_data()
    
    # 计算段长度
    n_channels1, n_samples1 = data1.shape
    n_channels2, n_samples2 = data2.shape
    segment_length1 = n_samples1 // n_segments
    segment_length2 = n_samples2 // n_segments
    
    # 为连接性图像创建子目录
    conn_dir = os.path.join(output_dir, "connectivity")
    os.makedirs(conn_dir, exist_ok=True)
    
    # 创建标准化器
    scaler = MinMaxScaler()
    
    print(f"数据总长度: 被试1={n_samples1}个样本点, 被试2={n_samples2}个样本点")
    print(f"每段长度: 被试1={segment_length1}个样本点, 被试2={segment_length2}个样本点")
    
    # 估算每段实际时间长度（秒）
    time_per_segment1 = segment_length1 / raw1.info['sfreq']
    time_per_segment2 = segment_length2 / raw2.info['sfreq']
    print(f"每段时间: 被试1={time_per_segment1:.2f}秒, 被试2={time_per_segment2:.2f}秒")
    
    for i in range(n_segments):
        # 处理每个段
        start1 = i * segment_length1
        end1 = (i + 1) * segment_length1 if i < n_segments - 1 else n_samples1
        seg_data1 = data1[:, start1:end1]
        
        start2 = i * segment_length2
        end2 = (i + 1) * segment_length2 if i < n_segments - 1 else n_samples2
        seg_data2 = data2[:, start2:end2]
        
        # 计算相关矩阵
        corr1 = np.corrcoef(seg_data1)
        corr2 = np.corrcoef(seg_data2)
        
        # 创建合并的成对连接性矩阵
        combined_channels = n_channels1 + n_channels2
        combined_corr = np.zeros((combined_channels, combined_channels))
        
        # 填充第一个被试数据
        combined_corr[:n_channels1, :n_channels1] = corr1
        
        # 填充第二个被试数据
        combined_corr[n_channels1:, n_channels1:] = corr2
        
        # 计算两个被试间的相关性
        cross_corr = np.zeros((n_channels1, n_channels2))
        
        # 如果两个段的长度不同，则将其调整为相同长度
        min_length = min(seg_data1.shape[1], seg_data2.shape[1])
        resized_data1 = seg_data1[:, :min_length]
        resized_data2 = seg_data2[:, :min_length]
        
        # 计算通道间相关性
        for ch1 in range(n_channels1):
            for ch2 in range(n_channels2):
                corr = np.corrcoef(resized_data1[ch1], resized_data2[ch2])[0, 1]
                cross_corr[ch1, ch2] = corr
        
        # 填充跨被试相关性
        combined_corr[:n_channels1, n_channels1:] = cross_corr
        combined_corr[n_channels1:, :n_channels1] = cross_corr.T
        
        # 创建图表
        plt.figure(figsize=(10, 8), dpi=100)
        plt.imshow(combined_corr, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        
        # 添加分界线
        plt.axhline(y=n_channels1-0.5, color='k', linestyle='-', alpha=0.5)
        plt.axvline(x=n_channels1-0.5, color='k', linestyle='-', alpha=0.5)
        
        # 添加标签
        plt.title(f"{subject_pair_name} - Inter/Intra Brain Connectivity - Segment {i+1}/{n_segments}")
        plt.xlabel("Channels")
        plt.ylabel("Channels")
        
        # 添加块标签
        mid_ch1 = n_channels1 // 2
        mid_ch2 = n_channels1 + n_channels2 // 2
        plt.text(mid_ch1, -1, "Subject 1", ha='center')
        plt.text(mid_ch2, -1, "Subject 2", ha='center')
        plt.text(-1, mid_ch1, "Subject 1", va='center', rotation=90)
        plt.text(-1, mid_ch2, "Subject 2", va='center', rotation=90)
        
        plt.tight_layout()
        
        # 保存图像
        output_file = os.path.join(conn_dir, f"{subject_pair_name}_connectivity_seg{i+1}.jpg")
        plt.savefig(output_file)
        plt.close()
        print(f"保存: {output_file}")

def process_subject_pair(subject1_file, subject2_file, output_dir, subject_pair_name):
    """处理一对被试的脑电数据，并生成图像"""
    print(f"\n开始处理被试对: {subject_pair_name}")
    
    try:
        # 加载脑电数据
        raw1 = load_eeg_data(subject1_file)
        raw2 = load_eeg_data(subject2_file)
        
        # 创建图像，增加到20个时间段
        create_eeg_images(raw1, raw2, output_dir, subject_pair_name, n_segments=20)
        
        print(f"被试对 {subject_pair_name} 处理完成\n")
        return True
    except Exception as e:
        print(f"处理被试对 {subject_pair_name} 时出错: {e}")
        return False

def find_subject_pairs(data_dir, pattern_type="sub"):
    """查找所有匹配的被试数据对，支持不同命名格式
    
    参数:
        data_dir: 数据目录
        pattern_type: 'hw' 表示H*/W*命名格式，'sub' 表示sub*_A/sub*_B命名格式
    """
    if pattern_type.lower() == "hw":
        # H*.vhdr和W*.vhdr格式
        h_files = sorted(glob.glob(os.path.join(data_dir, "H*.[vfst][hae][dtt]*")))  # 支持.vhdr、.set、.fat
        subject_pairs = []
        
        for h_file in h_files:
            # 从H文件中提取数字
            basename = os.path.basename(h_file)
            match = re.search(r'H(\d+)', basename)
            if not match:
                continue
                
            subject_num = match.group(1)
            file_ext = os.path.splitext(h_file)[1]  # 获取文件扩展名
            
            # 查找对应的W文件 (尝试相同扩展名)
            w_file = os.path.join(os.path.dirname(h_file), f"W{subject_num}{file_ext}")
            if os.path.exists(w_file):
                pair_name = f"Couple{subject_num.zfill(2)}_HW"
                subject_pairs.append((h_file, w_file, pair_name))
        
    elif pattern_type.lower() == "sub":
        # sub*_A.set/sub*_B.set 格式
        a_files = sorted(glob.glob(os.path.join(data_dir, "sub*_A.[sft][eah][tt]*")))  # 支持.set和.fat
        subject_pairs = []
        
        for a_file in a_files:
            # 从A文件中提取编号
            basename = os.path.basename(a_file)
            match = re.search(r'sub(\d+)_A', basename)
            if not match:
                continue
                
            subject_num = match.group(1)
            file_ext = os.path.splitext(a_file)[1]  # 获取文件扩展名
            
            # 查找对应的B文件 (尝试相同扩展名)
            b_file = os.path.join(os.path.dirname(a_file), f"sub{subject_num}_B{file_ext}")
            if os.path.exists(b_file):
                pair_name = f"Pair{subject_num.zfill(2)}_AB"
                subject_pairs.append((a_file, b_file, pair_name))
    else:
        print(f"不支持的模式类型: {pattern_type}")
        return []
    
    return subject_pairs

def main():
    """主函数 - 批量处理所有被试数据对"""
    start_time = time.time()
    
    try:
        # 在这里选择数据格式和目录
        pattern_type = "sub"  # 使用 "sub" 表示 sub01_A 格式，使用 "hw" 表示 H1/W1 格式
        
        # 设置目录路径 - 根据您选择的格式设置相应的目录
        if pattern_type == "sub":
            # 处理sub01_A/sub01_B格式的文件
            data_dir = "D:\\BaiduNetdiskDownload\\New_Data\\静息态"
            output_base_dir = "D:\\BaiduNetdiskDownload\\picture\\new_stranger_images"
        else:
            # 处理H1/W1格式的文件
            data_dir = "D:\\BaiduNetdiskDownload\\old_data_view"
            output_base_dir = "D:\\BaiduNetdiskDownload\\picture\\new_couple_images"
        
        # 创建主输出目录
        os.makedirs(output_base_dir, exist_ok=True)
        
        # 查找所有被试数据对
        subject_pairs = find_subject_pairs(data_dir, pattern_type)
        total_pairs = len(subject_pairs)
        
        print("="*70)
        print(f"找到 {total_pairs} 对被试数据")
        print("="*70)
        
        # 跟踪成功处理的数量
        successful_pairs = 0
        
        # 处理每一对被试数据
        for i, (file1, file2, pair_name) in enumerate(subject_pairs, 1):
            pair_output_dir = os.path.join(output_base_dir, pair_name)
            
            print(f"\n[{i}/{total_pairs}] 处理被试对: {pair_name}")
            print(f"被试1: {file1}")
            print(f"被试2: {file2}")
            print(f"输出目录: {pair_output_dir}")
            
            # 处理被试对
            success = process_subject_pair(file1, file2, pair_output_dir, pair_name)
            
            if success:
                successful_pairs += 1
                
            # 显示进度
            elapsed_time = time.time() - start_time
            avg_time_per_pair = elapsed_time / i
            est_time_remaining = avg_time_per_pair * (total_pairs - i)
            
            print(f"\n进度: {i}/{total_pairs} ({i/total_pairs*100:.1f}%)")
            print(f"已用时间: {elapsed_time/60:.1f}分钟")
            print(f"预计剩余时间: {est_time_remaining/60:.1f}分钟")
            print("-"*50)
        
        # 显示最终结果
        total_time = time.time() - start_time
        print("\n"+"="*70)
        print(f"处理完成! 总耗时: {total_time/60:.1f}分钟")
        print(f"成功处理: {successful_pairs}/{total_pairs} 对被试数据")
        print(f"所有生成的连通性图像可在以下目录找到: {output_base_dir}")
        print("="*70)
        
    except Exception as e:
        print(f"程序运行出错: {e}")

if __name__ == "__main__":
    main()