import open3d as o3d
import argparse
import os

def convert_ply_to_pcd(ply_path, pcd_path=None):
    try:
        # 使用 Tensor API 加载（兼容 CUDA）
        pcd_tensor = o3d.t.io.read_point_cloud(ply_path)

        # 转成 legacy 格式以支持保存
        pcd = pcd_tensor.to_legacy()

        if pcd.is_empty():
            print("Error: Loaded point cloud is empty.")
            return False

        # 设置默认保存路径
        if pcd_path is None:
            pcd_path = os.path.splitext(ply_path)[0] + ".pcd"

        # 保存为 PCD 格式
        success = o3d.io.write_point_cloud(pcd_path, pcd)
        if success:
            print(f"Successfully saved PCD to: {pcd_path}")
        else:
            print("Failed to write PCD file.")
        return success

    except Exception as e:
        print(f"[Exception] {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply_path", type=str, help="Path to input PLY file")
    parser.add_argument("--pcd_path", type=str, default=None, help="Optional output PCD file path")
    args = parser.parse_args()

    convert_ply_to_pcd(args.ply_path, args.pcd_path)

if __name__ == "__main__":
    main()

