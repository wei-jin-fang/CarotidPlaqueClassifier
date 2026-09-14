import os
import cv2
import pandas as pd
import numpy as np

def process_carotid_artery_crops():
    # 路径配置
    excel_path = '/media/data/wjf/data/Carotid_artery_excel.xlsx'
    mask_dir = '/media/data/wjf/data/Carotid_artery_mask'
    src_dir = '/media/data/wjf/data/Carotid_artery'
    dst_dir = '/media/data/wjf/data/Carotid_artery_crop_old'

    # 确保输出总目录存在
    os.makedirs(dst_dir, exist_ok=True)

    # 1. 读取Excel
    print(f"正在读取 Excel 文件: {excel_path}")
    df = pd.read_excel(excel_path)
    
    # 假设列名叫做 'name'，遍历每一个名字
    for index, row in df.iterrows():
        full_name = str(row['name']).strip()
        if not full_name or pd.isna(full_name):
            continue
            
        # 解析姓名和方向
        try:
            person_name, side = full_name.rsplit('_', 1)
        except ValueError:
            print(f"[{full_name}] 格式不正确，跳过。")
            continue
            
        # 2. 检查是否已经存在该文件夹，存在则跳过
        target_dir = os.path.join(dst_dir, full_name)
        if os.path.exists(target_dir):
            print(f"[{full_name}] 文件夹已存在，直接跳过。")
            continue
            
        # 3. 寻找对应的 mask 文件
        mask_path = os.path.join(mask_dir, f"{person_name}_mask.png")
        if not os.path.exists(mask_path):
            print(f"[{full_name}] 找不到对应的 mask 文件 ({mask_path})，跳过。")
            continue
            
        # 读取 mask 并寻找白色矩形框 (72x72)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # 二值化，确保纯粹的黑白
        _, thresh = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
        # 寻找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 获取所有矩形的外接框 (x, y, w, h)
        rects = [cv2.boundingRect(c) for c in contours]
        
        # 过滤掉噪点，保留宽高都在 65~80 左右的框 (容错处理，理论上应为 72x72)
        valid_rects = [r for r in rects if 65 <= r[2] <= 80 and 65 <= r[3] <= 80]
        
        if len(valid_rects) < 2:
            print(f"[{full_name}] 警告：在 mask 中找到的 72x72 矩形数量不足 2 个，跳过。")
            continue
            
        # 按 x 坐标从左到右排序
        valid_rects = sorted(valid_rects, key=lambda r: r[0])
        
        # 4. 关键逻辑：反向选择
        # 如果是 left，选择右边的框（索引 1，或最后一个）
        # 如果是 right，选择左边的框（索引 0）
        if side == 'left':
            target_rect = valid_rects[-1] # 右边的框
        elif side == 'right':
            target_rect = valid_rects[0]  # 左边的框
        else:
            print(f"[{full_name}] 后缀无法识别 ({side})，跳过。")
            continue
            
        x, y, w, h = target_rect
        
        # 5. 遍历该人员的原图目录，开始裁剪
        person_src_dir = os.path.join(src_dir, person_name)
        if not os.path.exists(person_src_dir):
            print(f"[{full_name}] 找不到对应的原图文件夹 ({person_src_dir})，跳过。")
            continue
            
        # 创建输出文件夹
        os.makedirs(target_dir, exist_ok=True)
        crop_count = 0
        
        # 使用 os.walk 遍历由于存在中间子文件夹（如数字字符串）的层级结构
        for root, dirs, files in os.walk(person_src_dir):
            for file in files:
                img_path = os.path.join(root, file)
                
                # 读取图像
                img = cv2.imread(img_path)
                if img is None:
                    # 如果不是图片文件，cv2.imread 会返回 None
                    continue
                    
                # 检查图像尺寸是否足够大，避免越界
                img_h, img_w = img.shape[:2]
                if y + h > img_h or x + w > img_w:
                    print(f"[{full_name}] 图像 {file} 尺寸小于 mask 框位置，跳过。")
                    continue
                
                # 剪裁图像 [y:y+h, x:x+w]
                cropped_img = img[y:y+h, x:x+w]
                
                # 保存图像，保持原名直接丢在文件夹里
                save_path = os.path.join(target_dir, file)
                # 如果遇到同名文件(比如不同子文件夹里都有 "5_JPG")，可以加上父文件夹名作前缀以防覆盖
                # 这里遵守要求：仍然以原来名称直接放在文件夹里面。如发现冲突，加点后缀防止覆盖
                if os.path.exists(save_path):
                    name_part, ext_part = os.path.splitext(file)
                    save_path = os.path.join(target_dir, f"{name_part}_{crop_count}{ext_part}")
                cv2.imwrite(save_path, cropped_img)
                crop_count += 1
                
        print(f"[{full_name}] 处理完成，共裁剪保存了 {crop_count} 张图片。")

if __name__ == '__main__':
    process_carotid_artery_crops()
    print("全部处理任务完成！")