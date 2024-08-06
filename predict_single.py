import argparse
import subprocess
from pathlib import Path
import os

import numpy as np
from skimage.io import imsave, imread
from tqdm import tqdm

from dataset.database import parse_database_name, get_ref_point_cloud
from estimator import name2estimator
from eval import visualize_intermediate_results
from prepare import video2image, prepare_image
from utils.base_utils import load_cfg, project_points
from utils.draw_utils import pts_range_to_bbox_pts, draw_bbox_3d
from utils.pose_utils import pnp


def weighted_pts(pts_list, weight_num=10, std_inv=10):
    """
    这个函数的目的是通过对最近的 weight_num 个点云应用加权平均来平滑或融合多个点云数据。最近的点云数据会有更高的权重，而较旧的点云数据的影响会被减小。
    """
    weights=np.exp(-(np.arange(weight_num)/std_inv)**2)[::-1] # wn
    pose_num=len(pts_list)
    if pose_num<weight_num:
        weights = weights[-pose_num:]
    else:
        pts_list = pts_list[-weight_num:]
    pts = np.sum(np.asarray(pts_list) * weights[:,None,None],0)/np.sum(weights)
    return pts

class EstimatorInterface:
    def __init__(self, cfg, database, iter=None, output_dir="out", transpose=False, image_size=640):
        self.iter = iter # iterations to refine the pose
        self.output_dir = output_dir
        self.database = database
        self.cfg = cfg
        self.image_size = image_size
        self.transpose = transpose

        if not os.path.exists(self.output_dir):
            print("creating output dir {}".format(self.output_dir))
            os.makedirs(self.output_dir)

        # self.args = args
        cfg = load_cfg(self.cfg)
        self.ref_database = parse_database_name(self.database)
        self.estimator = name2estimator[cfg['type']](cfg)
        self.estimator.build(self.ref_database, split_type='all')
        if self.iter:
            self.estimator.cfg['refine_iter'] = self.iter

        self.object_pts = get_ref_point_cloud(self.ref_database)
        self.object_bbox_3d = pts_range_to_bbox_pts(np.max(self.object_pts,0), np.min(self.object_pts,0))
    
    def predict(self, img_path):
        img = imread(img_path)
        img = prepare_image(img, self.image_size, self.transpose)
        h, w, _ = img.shape
        f=np.sqrt(h**2+w**2)
        K = np.asarray([[f,0,w/2],[0,f,h/2],[0,0,1]],np.float32)

         # we only refine one time after initialization
        pose_pr, inter_results = self.estimator.predict(img, K, pose_init=None) # 输出为当前相机位姿，世界到相机（在当前虚假K下，尺度为scale后的物体点云的尺度）

        pts, _ = project_points(self.object_bbox_3d, pose_pr, K) # 世界(点云文件)坐标系下的bbox，pose_pr: 世界到相机
        bbox_img = draw_bbox_3d(img, pts, (0,0,255))
        imsave(f'{str(self.output_dir)}/bbox.jpg', bbox_img)
        np.save(f'{str(self.output_dir)}/pose.npy', pose_pr)

        return pose_pr


def main(args):
    cfg = load_cfg(args.cfg)
    ref_database = parse_database_name(args.database)
    estimator = name2estimator[cfg['type']](cfg)
    estimator.build(ref_database, split_type='all')

    object_pts = get_ref_point_cloud(ref_database)
    object_bbox_3d = pts_range_to_bbox_pts(np.max(object_pts,0), np.min(object_pts,0))

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    (output_dir / 'images_raw').mkdir(exist_ok=True, parents=True)
    (output_dir / 'images_out').mkdir(exist_ok=True, parents=True)
    (output_dir / 'images_inter').mkdir(exist_ok=True, parents=True)
    (output_dir / 'images_out_smooth').mkdir(exist_ok=True, parents=True)

    que_num = video2image(args.video, output_dir/'images_raw', 1, args.resolution, args.transpose)

    pose_init = None
    hist_pts = []
    for que_id in tqdm(range(que_num)):
        img = imread(str(output_dir/'images_raw'/f'frame{que_id}.jpg'))
        # generate a pseudo K
        h, w, _ = img.shape
        f=np.sqrt(h**2+w**2)
        K = np.asarray([[f,0,w/2],[0,f,h/2],[0,0,1]],np.float32)

        if pose_init is not None:
            estimator.cfg['refine_iter'] = 1 # we only refine one time after initialization
        pose_pr, inter_results = estimator.predict(img, K, pose_init=pose_init) # 输出为当前相机位姿，世界到相机（在当前虚假K下，尺度为scale后的物体点云的尺度）
        pose_init = pose_pr

        pts, _ = project_points(object_bbox_3d, pose_pr, K) # 世界(点云文件)坐标系下的bbox，pose_pr: 世界到相机
        bbox_img = draw_bbox_3d(img, pts, (0,0,255))
        imsave(f'{str(output_dir)}/images_out/{que_id}-bbox.jpg', bbox_img)
        np.save(f'{str(output_dir)}/images_out/{que_id}-pose.npy', pose_pr)
        imsave(f'{str(output_dir)}/images_inter/{que_id}.jpg', visualize_intermediate_results(img, K, inter_results, estimator.ref_info, object_bbox_3d))

        hist_pts.append(pts)
        pts_ = weighted_pts(hist_pts, weight_num=args.num, std_inv=args.std) # 对box点云历史加权平均，得到smooth输出
        pose_ = pnp(object_bbox_3d, pts_, K)
        pts__, _ = project_points(object_bbox_3d, pose_, K)
        bbox_img_ = draw_bbox_3d(img, pts__, (0,0,255))
        imsave(f'{str(output_dir)}/images_out_smooth/{que_id}-bbox.jpg', bbox_img_)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/gen6d_pretrain_singleframe.yaml')
    parser.add_argument('--database', type=str, default="custom/mycup")
    parser.add_argument('--output', type=str, default="single_image_out2")
    parser.add_argument('--iters', type=int, default=1, help="times of refinement iterations")

    # input video process
    parser.add_argument('--transpose', action='store_true', dest='transpose', default=False)
    parser.add_argument('--image_size', type=int, default=640)

    args = parser.parse_args()

    # image_path = "example_images/ceshitu.png"
    # image_path = "single_image_out/example_images/1022764463.jpg"
    # image_path = "/home/junpeng.hu/Documents/ws_gen6d/Gen6D/data/custom/mycup/testnewmeta/images_raw/frame0.jpg"
    # image_path = "single_image_out/example_images/5301152175403164541.jpg"
    # args.output = "single_image_out3"

    # image_path = "single_image/example_images/5301152175403164554.jpg"
    # args.output = "single_image/single_image_out4"

    # image_path = "single_image/example_images/5301152175403164559.jpg"
    # args.output = "single_image/single_image_out5"

    # image_path = "single_image/example_images/5301152175403164561.jpg"
    # args.output = "single_image/single_image_out6"

    # image_path = "single_image/example_images/5301152175403164561.jpg"
    # args.output = "single_image/single_image_out7"
    # args.iters = 8

    # image_path = "/home/junpeng.hu/Documents/ws_gen6d/Gen6D/single_image/example_images/1022764463.jpg"
    # args.output = "single_image/single_image_out8"
    # args.iters = 8

    # image_path = "single_image/example_images/5301152175403164593.jpg"
    # args.output = "single_image/single_image_out9"
    # args.iters = 8

    # image_path = "single_image/example_images/5301152175403164593.jpg"
    # args.output = "single_image/single_image_out10" # with crop
    # args.iters = 8

    # image_path = "single_image/example_images/ceshitu2.jpg"
    # args.output = "single_image/single_image_out11" # with crop
    # args.iters = 8

    image_path = "/home/junpeng.hu/Documents/ws_gen6d/Gen6D/single_image/example_images/ceshitu.png"
    args.output = "single_image/single_image_out12" # with crop
    args.iters = 8
    
    interface = EstimatorInterface(args.cfg, args.database, output_dir=args.output, transpose=args.transpose, iter=args.iters, image_size=args.image_size)

    pose_pred = interface.predict(image_path) # 世界到相机pose，世界坐标系定义为interface.object_pts的世界坐标系
    print(pose_pred)