import cv2
import numpy as np
from rembg import remove
from PIL import Image
import io

def remove_background_to_white(input_path, output_path):
    """
    去除图片背景并设置为白色
    
    参数:
        input_path (str): 输入图片路径
        output_path (str): 输出图片路径
    """
    # 读取输入图片
    with open(input_path, 'rb') as f:
        input_image = f.read()
    
    # 使用rembg去除背景
    output_image = remove(input_image)
    
    # 将输出转换为PIL Image
    image = Image.open(io.BytesIO(output_image)).convert("RGBA")
    
    # 创建一个白色背景
    white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    
    # 合并图像（将透明部分替换为白色）
    result = Image.alpha_composite(white_bg, image)
    
    # 转换为RGB模式（去除alpha通道）
    result = result.convert("RGB")
    
    # 保存结果
    result.save(output_path, "JPEG", quality=95)
    print(f"处理完成，结果已保存到: {output_path}")

if __name__ == "__main__":
    input_image = "chasis.png"  # 替换为你的输入图片路径
    output_image = "output.png"  # 替换为你想要的输出路径
    
    remove_background_to_white(input_image, output_image)
