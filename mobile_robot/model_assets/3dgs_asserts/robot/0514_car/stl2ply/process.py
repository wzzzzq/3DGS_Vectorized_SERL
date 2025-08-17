import open3d as o3d
import os
import numpy as np

def convert_mesh_to_pointcloud(input_folder, output_folder=None):
    """
    将目录中所有PLY网格文件转换为纯点云格式
    :param input_folder: 输入目录路径
    :param output_folder: 输出目录路径（默认覆盖原文件）
    """
    if output_folder is None:
        output_folder = input_folder
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.ply'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                # 尝试读取为点云
                pcd = o3d.io.read_point_cloud(input_path)
                if not pcd.has_points():
                    # 如果失败则读取为网格并提取顶点
                    mesh = o3d.io.read_triangle_mesh(input_path)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = mesh.vertices
                    
                    # 保留颜色（如果有）
                    if mesh.vertex_colors:
                        pcd.colors = mesh.vertex_colors
                
                # 保存为CloudCompare兼容格式
                o3d.io.write_point_cloud(
                    output_path,
                    pcd,
                    write_ascii=False,  # 二进制格式
                    compressed=True,
                    print_progress=True
                )
                print(f"Converted {filename} successfully")
                
            except Exception as e:
                print(f"Failed to convert {filename}: {str(e)}")

# 使用示例
convert_mesh_to_pointcloud('/home/cfy/cfy/cfy/lerobot_nn/mobile_ai_rl/mobile_ai_model/stl2ply')
