import open3d as o3d
import numpy as np
import os

def draw_cameras(vis, poses, scale=0.2):
    """
    Draw camera poses in Open3D visualization.

    Args:
        vis (o3d.visualization.Visualizer): Open3D Visualizer instance.
        poses (list of np.ndarray): List of 4x4 camera poses.
        scale (float): Scale of the camera frustum.
    """
    for pose in poses:
        # Define the frustum points in the camera coordinate system
        frustum_points = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [1, -1, 1],
            [-1, -1, 1],
            [-1, 1, 1]
        ]) * scale

        # Transform frustum points to world coordinates
        frustum_points = (pose @ np.vstack((frustum_points.T, np.ones((1, frustum_points.shape[0]))))).T[:, :3]

        # Create lines for the frustum
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 2], [2, 3], [3, 4], [4, 1]
        ]
        
        # Create an Open3D line set for the frustum
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(frustum_points),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.paint_uniform_color([1, 0, 0])  # Red color for camera frustums
        vis.add_geometry(line_set)

def main():
    # Load PLY file
    ply_file_path = "path/to/your/point_cloud.ply"
    pcd = o3d.io.read_point_cloud(ply_file_path)

    # Define camera poses (4x4 transformation matrices)
    poses = [
    ]

    pose_root = "/home/junpeng.hu/Documents/ws_gen6d/Gen6D/data/custom/mouse/test3/images_out"

    for f in os.listdir(pose_root):
        if not (f.endswith(".npy")):
            continue
        path = os.path.join(pose_root,f)
        pose = np.load(path)
        h_pose = np.eye(4)
        h_pose[:3,:]=pose
        poses.append(h_pose)
    
    print(poses)

    # # Create visualizer
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()

    # # Add point cloud to visualizer
    # vis.add_geometry(pcd)

    # # Draw camera poses
    # draw_cameras(vis, poses)

    # # Run visualizer
    # vis.run()
    # vis.destroy_window()

if __name__ == "__main__":
    main()
