import argparse
import os
import cv2
import torch
import numpy as np
import open3d as o3d
import scipy.ndimage
import matplotlib.pyplot as plt 

# from transformers import CLIPTokenizer, CLIPModel
# from openpoints.dataset.data_util import crop_pc
# from inference import load_model

from openpoints.transforms import build_transforms_from_cfg
from util import save_pickle, load_pickle, load_easyconfig_from_yaml
from vis_exec import visualization_exec

from tool_repos.FastSAM.fastsam import FastSAM
from fastsam_prompt import FastSAMPrompt
from PIL import Image

from embodied_analogy.utility.constants import *
from embodied_analogy.representation.basic_structure import Frame
from embodied_analogy.environment.manipulate_env import ManipulateEnv
from embodied_analogy.utility.estimation.coarse_joint_est import coarse_t_from_tracks_3d, coarse_R_from_tracks_3d, coarse_estimation
from embodied_analogy.utility.utils import visualize_pc, set_random_seed

set_random_seed(ALGO_SEED)


class KPSTExecutor(object):
    def __init__(self, args, cfg):
        if torch.cuda.is_available() is False:
            raise ValueError("Please use GPU for KPST Executor.")
        device = 'cuda'
        self.device = device

        from transformers import CLIPTokenizer, CLIPModel
        
        # clip_model = None
        clip_model = {
            "tokenizer": CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32"),
            "model": CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        }
        self.clip_model = clip_model
        print("Finish Loading CLIP Model.")

        # self.sam_model = FastSAM('./tool_repos/FastSAM/weights/FastSAM-X.pt')
        self.sam_model = FastSAM(os.path.join(ASSET_PATH, 'ckpts/fastSAM/FastSAM-X.pt')) 
        self.sam_prompt_model = FastSAMPrompt(device=device)

        self.camera = None
        self.args = args
        self.cfg = cfg
        self.device = device
        self.desc = 'None'
        self.desc_feat = None    # (1, 512)

        args.not_load = False  
        self.kpst_model = None
        self.unit_crop_r = None
        self.data_transform = build_transforms_from_cfg('test', cfg.datatransforms)
        self.set_kpst_model(args.pretrained_path)
        print("Finish Loading KPST Model.")
    
    def set_kpst_model(self, pretrained_path):
        self.args.cfg = '/'.join(pretrained_path.split('/')[:-2] + ['cfg.yaml']) 
        cfg = load_easyconfig_from_yaml(self.args.cfg)
        self.print_kpst_model()
        if cfg.seed is None: cfg.seed = 0
        self.args.pretrained_path = pretrained_path
        from inference import load_model
        self.kpst_model = load_model(self.args, cfg)
        self.kpst_model.to(self.device)
        self.kpst_model.eval()
        self.unit_crop_r = cfg.dataset.common.get('unit_r', None)
        self.args.voxel_max = cfg.dataset.test.voxel_max
        self.cfg = cfg 
        # Notice: voxel_size is not set by cfg, you need to run self.set_pcd_voxel_size() mannually.  

    def print_kpst_model(self):
        print(f"cfg_file_path={self.args.cfg}")
    
    def set_desc(self, desc):

        # self.desc = 'None'
        # self.desc_feat = np.random.randn(512)

        # clip_open_safe_fp = 'results/exec_save/clip_feat'
        # clip_open_safe_fp = os.path.join(clip_open_safe_fp, desc+'.npy')
        # if not os.path.exists(clip_open_safe_fp):
        #     os.makedirs('results/exec_save/clip_feat', exist_ok=True)
        desc = ' '.join(desc.split('_'))
        self.desc = desc
        inputs = self.clip_model['tokenizer'](desc, padding=True, return_tensors="pt")
        text_features = self.clip_model['model'].get_text_features(**inputs)             # (1, 512)
        text_features = text_features.detach().numpy().reshape(-1)                       # (512)
        self.desc_feat = text_features
        #     np.save(clip_open_safe_fp, self.desc_feat)
        # else:
        #     self.desc = desc
        #     print("load pre_clip_feature!")
        #     self.desc_feat = np.load(clip_open_safe_fp)

        print(f"Set Desc: {desc}.")

    def set_camera(self, camera, H=0, W=0):
        if isinstance(camera, o3d.camera.PinholeCameraIntrinsic):
            self.camera = camera
        else:
            if (H == 0) or (W == 0):
                raise ValueError(f"When set camera mannually, H != 0 and W != 0, but get (H,W)=({H},{W}).")
            self.camera = o3d.camera.PinholeCameraIntrinsic()
            self.camera.set_intrinsics(W, H, camera[0,0], camera[1,1], camera[0,2], camera[1,2])

    @staticmethod
    def display_and_capture_points(image, title='Default'):
        image_pil = Image.fromarray(np.uint8(image))
        points = []

        def onclick(event):
            ix, iy = event.xdata, event.ydata
            print(f'Point: ({ix}, {iy})')
            ax.plot(ix, iy, 'ro')
            fig.canvas.draw()
            points.append((ix, iy))
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.imshow(image_pil)
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        fig.canvas.mpl_disconnect(cid)
        points = np.array(points)        # (N, 2), (w, h)-format
        return points
    
    def segment_robot_body(self, image, robot_anchor, robot_anchor_label, vis_dir=None, commit=''):
        # robot_anchor: (N, 2), (h,w)-format, numpy

        # robot_anchor = np.array([[935, 1042]])
        # robot_anchor_label = np.ones(robot_anchor.shape[0]).astype(np.uint32)

        if robot_anchor is None:
            robot_anchor = KPSTExecutor.display_and_capture_points(image, title='robot mask')
            robot_anchor = [[int(x[1]), int(x[0])] for x in robot_anchor]   # (w, h) --> (h, w)
            robot_anchor = np.array(robot_anchor)
            robot_anchor_label = np.ones(robot_anchor.shape[0]).astype(np.uint32)

        ra = np.concatenate([robot_anchor[:, 1:2].copy(), robot_anchor[:, 0:1].copy()], axis=-1) # (h, w) --> (w, h)

        if self.args.seg_downsample_ratio > 1:
            img_pil = Image.fromarray(image)
            new_height = img_pil.height // self.args.seg_downsample_ratio
            new_width = img_pil.width // self.args.seg_downsample_ratio
            resized_img = np.asarray(img_pil.resize((new_width, new_height)))
            ra = ra // self.args.seg_downsample_ratio
        else:
            resized_img = image

        everything_results = self.sam_model(resized_img, device=self.device, retina_masks=True, imgsz=1024, 
                                            conf=0.4, iou=0.9,)
        self.sam_prompt_model.set_image_result(resized_img, everything_results)
        mask = self.sam_prompt_model.point_prompt(ra, robot_anchor_label)
        
        if self.args.seg_downsample_ratio > 1:  
            mask = scipy.ndimage.zoom(mask, (1, self.args.seg_downsample_ratio, self.args.seg_downsample_ratio), order=0)

        if vis_dir is not None:
            if len(commit) > 0: commit = '_' + commit
            desc_id = self.desc.replace(' ', '_')
            self.sam_prompt_model.img = image
            save_fp = os.path.join(vis_dir, f'{desc_id}'+commit+'_'+f'mask.png')
            self.sam_prompt_model.plot(annotations=mask, output_path=save_fp)

        return mask[0, :, :]         # (H, W)

    def _get_intrinsic_parameter(self):
        cx = self.camera.intrinsic_matrix[0, 2]
        cy = self.camera.intrinsic_matrix[1, 2]
        fx = self.camera.intrinsic_matrix[0, 0]
        fy = self.camera.intrinsic_matrix[1, 1]
        return cx, cy, fx, fy

    def find_corresponding_3d_point(self, gripper_2d_pos, pcd_scene, depth_image):
        # pdb.set_trace()
        # gripper_2d_pos: (h,w)-format
        gripper_2d_pos = gripper_2d_pos.astype(np.int32)
        depth_value = depth_image[gripper_2d_pos[0], gripper_2d_pos[1]]

        y, x = gripper_2d_pos
        if depth_value == 0:
            return None
        z = depth_value / 1000.0
        cx, cy, fx, fy = self._get_intrinsic_parameter()
        point_3d = np.array([(x - cx) * z / fx, (y - cy) * z / fy, z])
        pcd_points = np.asarray(pcd_scene.points)
        distances = np.linalg.norm(pcd_points - point_3d, axis=1)
        closest_point_index = np.argmin(distances)
        closest_point = pcd_points[closest_point_index]

        return closest_point

    def set_pcd_voxel_size(self, voxel_size):
        self.args.voxel_size = voxel_size

    def pcd_cut(self, pcd, area_bound):
        # pdb.set_trace()
        pcd_pos = np.asarray(pcd.points)
        is_available = (pcd_pos[:, 0] > area_bound[0]) & (pcd_pos[:, 0] < area_bound[1]) & \
                        (pcd_pos[:, 1] > area_bound[2]) & (pcd_pos[:, 1] < area_bound[3]) & \
                        (pcd_pos[:, 2] > area_bound[4]) & (pcd_pos[:, 2] < area_bound[5]) 
        pcd_cut = o3d.geometry.PointCloud()
        pcd_cut.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[is_available])
        pcd_cut.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[is_available])
        return pcd_cut

    def geometric_generation(self, rgb_image, depth_image, gripper_pos, mask=None, area_bound=None):
        """
        NOTE: 这里的 depth_image 的尺度是 mm
        """
        # pdb.set_trace()
        color_raw = o3d.geometry.Image(rgb_image[:, :, :3])      # (H, W, C)
        depth_scene = depth_image.copy()                          
        if mask is not None: 
            depth_scene[mask] = 0
        depth_scene = o3d.geometry.Image(depth_scene)             # (H, W)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_scene, convert_rgb_to_intensity=False)
        pcd_scene = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.camera)
        if len(gripper_pos) == 3:
            gripper_3d_pos = gripper_pos
        else:
            gripper_3d_pos = self.find_corresponding_3d_point(gripper_pos, pcd_scene, depth_image)
            if gripper_3d_pos is None:
                raise ValueError("gripper_2d_pos={gripper_pos}, but depth is broken, can not find gripper_3d_pos")
        
        if area_bound is not None:
            pcd_scene = self.pcd_cut(pcd_scene, area_bound)

        voxel_down_pcd = pcd_scene.voxel_down_sample(voxel_size=self.args.voxel_size)

        if self.unit_crop_r is not None:
            # pdb.set_trace()
            coord, feat = np.asarray(voxel_down_pcd.points), np.asarray(voxel_down_pcd.colors)
            is_available = (coord[:, 0] > gripper_3d_pos[0] - self.unit_crop_r) & \
                           (coord[:, 0] < gripper_3d_pos[0] + self.unit_crop_r) & \
                           (coord[:, 1] > gripper_3d_pos[1] - self.unit_crop_r) & \
                           (coord[:, 1] < gripper_3d_pos[1] + self.unit_crop_r) & \
                           (coord[:, 2] > gripper_3d_pos[2] - self.unit_crop_r) & \
                           (coord[:, 2] < gripper_3d_pos[2] + self.unit_crop_r)
            coord, feat = coord[is_available], feat[is_available]
            voxel_down_pcd = o3d.geometry.PointCloud()
            voxel_down_pcd.points = o3d.utility.Vector3dVector(coord)
            voxel_down_pcd.colors = o3d.utility.Vector3dVector(feat)

        return voxel_down_pcd, gripper_3d_pos, pcd_scene
    
    def get_kps_3d(self, pcd, gripper_3d_pos, radius=0.08, kps_max=256):
        """
        筛选出 pcd 中距离 gripper_3d_pos 一定范围内的点
        """
        kps = np.array(pcd.points)
        dist = np.linalg.norm(kps - gripper_3d_pos, axis=1)
        idx = dist < radius
        kps, dist = kps[idx], dist[idx]
        if kps.shape[0]> kps_max:
            idx = np.argsort(dist)[:kps_max]
            kps, dist = kps[idx], dist[idx]

        weights = 1 / (dist + self.args.weight_beta)

        if kps.shape[0] < kps_max:
            # pdb.set_trace()
            repeat_idx = np.random.choice(kps.shape[0], size=kps_max - kps.shape[0], replace=True)
            kps = np.concatenate([kps, kps[repeat_idx]], axis=0)
            weights = np.concatenate([weights, weights[repeat_idx]])

        return kps, weights
    
    def get_kpst_model_prediction(self, data, return_np=False, inference_num=20):
        # pdb.set_trace()
        # data = {'pos': coord, 'x': feat, 'dtraj': qry_pos, 'text_feat': text_features}, Tensor
        pos, feat = data['pos'], data['x']                         
        feat = torch.concat([feat, pos], axis=-1)                     
        dtraj, text_feat = data['dtraj'], data['text_feat']                       # (Q, T=5, 3)        
        query_np = dtraj[:, 0, :]  
                 
        pos = pos.unsqueeze(0).to(self.device).float()                      # (1, N, 3)
        feat = feat.unsqueeze(0).to(self.device).float()                    # (1, N, 6)
        query = query_np.unsqueeze(0).to(self.device).float()               # (1, Q, 3)
        text_feat = text_feat.unsqueeze(0).to(self.device).float()          # (1, Ft)

        traj_prediction = self.kpst_model.inference(pos, feat, text_feat, query, num_sample=inference_num).squeeze(0)   # (Q, M, T-1, 3)
      
        traj_prediction = traj_prediction.transpose(0, 1)                     # (M, Q, T-1=3, 3)
        qry = query.unsqueeze(-2).repeat(traj_prediction.shape[0], 1, 1, 1)   # (M, Q, 1, 3)
        kpst = torch.cat([qry, traj_prediction], -2)                          # (M, Q, T=5, 3)

        # pdb.set_trace()
        dist = torch.mean(torch.sum(torch.norm(kpst[:, :, :-1] - kpst[:, :, 1:], dim=-1), dim=-1), dim=-1) 
        uid = torch.argmax(dist, dim=0)
        kpst = kpst[uid: uid+1]
        print(f"kpst_average_length = {dist[uid]}")

        if return_np is True:
            kpst = kpst.detach().cpu().numpy()
        return kpst
    
    def get_kpst_prediction(self, pcd, kps_3d, return_np=False):
        # pdb.set_trace()
        qry_pos = kps_3d[:, np.newaxis, :]   # (Q, 1, 3), = (Q, T, 3)
        pcd_coord = np.array(pcd.points)         # (N, 3)
        pcd_feat = np.array(pcd.colors)          # (N, 3)
        from openpoints.dataset.data_util import crop_pc
        coord, feat, _ = crop_pc(
            pcd_coord, pcd_feat, None, 'test', self.args.voxel_size, self.args.voxel_max, 
            variable=False, voxel_downsample_bar=0.02, 
            mask=None, mask_ratio=None)
        data = {
            'pos': coord,
            'x': feat,
            'dtraj': qry_pos,
            'text_feat': self.desc_feat
        }
        data = self.data_transform(data)
        # norm_coord = coord.mean(0)               # (3, ), numpy
        norm_coord = qry_pos.mean(0).mean(0)

        kpst = self.get_kpst_model_prediction(data, return_np=return_np)    # (M, Q, T, 3), Tensor
        
        if return_np is False:
            kpst = kpst + torch.from_numpy(norm_coord).to(self.device)      # (M, Q, T, 3)
            kpst = kpst.float()
        else:
            kpst = kpst + norm_coord[np.newaxis, np.newaxis, np.newaxis, :]  # (M, Q, T, 3)
        
        return kpst

    def rigid_transform_3d(self, A, B, weights=None, weight_threshold=0):
        # pdb.set_trace()
        """ 
        CodeBase: https://github.com/zhongcl-thu/3D-Implicit-Transporter
        Input:
            - A:       [bs, num_corr, 3], source point cloud
            - B:       [bs, num_corr, 3], target point cloud
            - weights: [bs, num_corr]     weight for each correspondence 
            - weight_threshold: float,    clips points with weight below threshold
            all is Tensor
        Output:
            - R, t 
        """
        # pdb.set_trace()
        bs = A.shape[0]
        if weights is None:
            weights = torch.ones_like(A[:, :, 0])
        weights[weights < weight_threshold] = 0
        # weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-6)
        # import pdb;pdb.set_trace()
        # find mean of point cloud
        centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
        centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)

        # subtract mean
        Am = A - centroid_A
        Bm = B - centroid_B

        # construct weight covariance matrix
        Weight = torch.diag_embed(weights)
        H = Am.permute(0, 2, 1) @ Weight @ Bm

        # find rotation
        try:
            U, S, Vt = torch.svd(H.cpu())
            U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
            delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
            eye = torch.eye(3)[None, :, :].repeat(bs, 1, 1).to(A.device)
            eye[:, -1, -1] = delta_UV
            R = Vt @ eye @ U.permute(0, 2, 1)
            t = centroid_B.permute(0,2,1) - R @ centroid_A.permute(0,2,1)
            # warp_A = transform(A, integrate_trans(R,t))
            # RMSE = torch.sum( (warp_A - B) ** 2, dim=-1).mean()
            return R, t, True
        except:
            print("Fail to Generation.")
            return torch.eye(3).unsqueeze(0).repeat(A.shape[0]).to(self.device), torch.zeros(A.shape[0], 3, 1).to(self.device), False

    def get_motion_planning(self, kpst, weights, plan_step=1, commit=''):
        """
        基于输入的关键点时间序列轨迹(kpst)和权重(weights)，计算一系列刚体变换(旋转和平移)，从而生成运动规划路径
        """
        if plan_step > kpst.shape[2]:
            raise ValueError(f"plan_step={plan_step} should be smaller than KPST.Length={kpst.shape[2]}")
        motion_plan = []
        if weights.ndim == 1:
            weights = weights[None, :].repeat(kpst.shape[0], 1)    # (M, Q)
        for i in range(plan_step):
            pcd_A = kpst[:, :, i]    # (M, Q, 3)
            pcd_B = kpst[:, :, i+1]    # (M, Q, 3)
            R, t, success = self.rigid_transform_3d(pcd_A, pcd_B, weights=weights)
            R = R.detach().cpu().numpy()
            t = t.detach().cpu().numpy()
            motion_plan.append((R, t, success))
        return motion_plan

    def kpst_motion_execusion(
        self,
        rgb_image,
        depth_image,
        gripper_pos, 
        area_bound=None,
        policy_radius=0.08,
        policy_env='voxel',
        policy_kps_max=256,
        robot_anchor=None, 
        robot_anchor_label=None,
        plan_step=1, 
        commit='',
        vis_dir=None
    ):
        """
            rgb_image: (H, W, 3), RGB, uint8.
            depth_image: (H, W), Depth, float32. (scale: meter)
            gripper_pos: (3, ), the 3d position of gripper. or (2, ) with (h,w)-format, the 2d position of the gripper.
            policy_radius: float, the radius for kps sampling & policy generation.
            policy_env: 'voxel' or 'origin', get kps from voxel-downsampling or origin point cloud.
            robot_anchor: (N, 2), (h,w)-format, numpy, Point or BBox SAM prompt for robot-body segmentation.
            plan_step: int, the step for close-loop motion planning.
            commit: str, the commit for saving the results.
        """
        print(f"KPST-MOTION EXECUSION, Description=[{self.desc}].")
        mask = self.segment_robot_body(rgb_image, robot_anchor, robot_anchor_label, commit=commit, vis_dir=vis_dir)
        # 获取三维点云和 contact_3d
        pcd, gripper_3d_pos, pcd_org = self.geometric_generation(rgb_image, depth_image, gripper_pos, mask=mask, area_bound=area_bound)

        # 获取 contact_3d 附近的点
        if policy_env == 'voxel':
            kps_3d, weights = self.get_kps_3d(pcd, gripper_3d_pos, radius=policy_radius, kps_max=policy_kps_max)
        else:
            kps_3d, weights = self.get_kps_3d(pcd_org, gripper_3d_pos, radius=policy_radius, kps_max=policy_kps_max)
            
        kpst = self.get_kpst_prediction(pcd, kps_3d, return_np=False)   # (M, Q, T, 3), cuda
        weights = torch.Tensor(weights / np.sum(weights)).to(self.device)
        motion_plan = self.get_motion_planning(kpst, weights, plan_step=plan_step, commit=commit)
        if vis_dir is not None:
            vis_kpst = kpst.detach().cpu().numpy()
            vis_result = self.save_exec_data_to_dir(vis_kpst, gripper_3d_pos, motion_plan, pcd_org, vis_dir=vis_dir, commit=commit)
        else:
            vis_result = None
        return motion_plan, vis_result
    
    def save_exec_data_to_dir(self, vis_kpst, gripper_3d_pos, motion_plan, pcd_org, vis_dir, commit=''):
        pcd_numpy = np.concatenate([pcd_org.points, pcd_org.colors], axis=-1)
        print(f"Number of Points: {pcd_numpy.shape[0]}")
        result = {
            'description': self.desc,
            'model': self.args.pretrained_path,
            'inference_num': 1,
            'traj_prediction': vis_kpst,                   # (M, Q, T, 3)
            'pcd': pcd_numpy,                              # (N, 6)
            'gripper_3d_pos': gripper_3d_pos,              # (3, )
            'motion_plan': motion_plan,                    # [(R, t, success), ...]
        }
        if len(commit) > 0:
            commit = '_' + commit
        # desc_id = self.desc.replace(' ', '_')
        save_fp = os.path.join(vis_dir, f'robot_exec.pkl')
        save_pickle(save_fp, result)
        return result
    
def get_generalFlow(
    frame, 
    instruction: str, 
    # save_dir: str,
    visualize=False
):
    """
    frame: Frame
    
    对于一个 Frame 提取 general_flow
    """
    parser = argparse.ArgumentParser('KPST Model Execusion')
    parser.add_argument('--voxel_size', type=float, default=0.01, help='the voxel size for downsampling the input point cloud')
    parser.add_argument('--voxel_max', type=int, default=2048)
    parser.add_argument('--weight_beta', type=float, default=0.1, help='the weight beta for the KPST model')
    parser.add_argument('--seg_downsample_ratio', type=int, default=1, help='the downsample ratio for robot-body segmentation')
    pretrained_path = os.path.join(ASSET_PATH, "ckpts/generalFlow/kpst_hoi4d/ScaleGFlow-B/checkpoint/ckpt_best_train_scalegflow_b.pth")
    parser.add_argument('-p', '--pretrained_path', type=str, default=pretrained_path) 
    
    args, opts = parser.parse_known_args()
    args.cfg = '/'.join(args.pretrained_path.split('/')[:-2] + ['cfg.yaml']) 
    # args.save_dir = save_dir
    cfg = load_easyconfig_from_yaml(args.cfg)
    # print(f"cfg_file_path={args.cfg}")
    cfg.update(opts)
    if cfg.seed is None: 
        cfg.seed = 0
    # os.makedirs(args.save_dir, exist_ok=True)
    print("Loading General Flow Model...")
    exec_model = KPSTExecutor(args, cfg)
    H, W = frame.rgb.shape[0], frame.rgb.shape[1]
    exec_model.set_camera(camera=frame.K, H=H, W=W)
    exec_model.set_desc(instruction)

    # (w, h) --> (h, w)
    contact_uv = frame.contact2d
    gripper_pos = np.array([contact_uv[1], contact_uv[0]])
    # 获取三维点云和 contact_3d
    pcd, gripper_3d_pos, pcd_org = exec_model.geometric_generation(
        rgb_image=frame.rgb,
        # 由于 general_flow 的 exec_model 使用的单位是 mm, 这里需要对于单位进行转换
        depth_image=frame.depth * 1000.0,
        gripper_pos=gripper_pos,
        mask=frame.robot_mask,
        area_bound=None
    )

    # 获取 contact_3d 附近的点, (Q, 3)
    kps_3d, weights = exec_model.get_kps_3d(
        pcd=pcd,
        gripper_3d_pos=gripper_3d_pos,
        radius=0.1,
        kps_max=128
    )
        
    # (1, Q, T, 3), cuda
    kpst = exec_model.get_kpst_prediction(
        pcd=pcd,
        kps_3d=kps_3d,
        return_np=False
    )   
    weights = torch.Tensor(weights / np.sum(weights)).to(exec_model.device)
    motion_plan = exec_model.get_motion_planning(kpst, weights, plan_step=3, commit="")
    
    if visualize:
        pcd_numpy = np.concatenate([pcd_org.points, pcd_org.colors], axis=-1)
        pcd_numpy[:, 3:] = pcd_numpy[:, 3:] * 255.0
        
        # print(f"Number of Points: {pcd_numpy.shape[0]}")
        data = {
            'description': instruction,
            'model': args.pretrained_path,
            'inference_num': 1,
            'traj_prediction': kpst.detach().cpu().numpy(),# (M, Q, T, 3)
            'pcd': pcd_numpy,                              # (N, 6)
            'gripper_3d_pos': gripper_3d_pos,              # (3, )
            'motion_plan': motion_plan,                    # [(R, t, success), ...]
        }
        
        parser = argparse.ArgumentParser('KPST visualization')
        parser.add_argument('-n', '--max_traj', type=int, default=48)
        args, opts = parser.parse_known_args()
        visualization_exec(args, data)
    
    kpst = kpst[0] # Q, T, 3
    return kpst.detach().cpu().numpy()

def test_get_generalFlow():  
    input_dir = "/home/zby/Programs/general_flow/demo/input/safe_0_hand"
    frame = Frame(
        rgb=cv2.imread(os.path.join(input_dir, 'rgb.jpg'), cv2.IMREAD_COLOR),
        depth=cv2.imread(os.path.join(input_dir, 'dep.png'), cv2.IMREAD_ANYDEPTH).astype(np.float32),
        K=np.load(input_dir + '/' + 'camera_in.npy')
    )
    frame.robot_mask = np.load(input_dir + '/' + 'hand_mask.npy')
    frame.contact2d = np.array([972, 658])
    get_generalFlow(
        frame=frame, 
        instruction="open_Safe", 
        visualize=True
    )


class GeneralFlow_ManipEnv(ManipulateEnv):
    def __init__(self, cfg):
        """
        这里的 cfg 来自于 embodied_analogy 测试的那些 cfg
        """
        self.cfg = cfg
        ManipulateEnv.__init__(self, cfg)
        
        # 清空来自 embodied analogy 的 obj_repr 中的 joint_dict
        self.obj_repr.coarse_joint_dict = None
        self.obj_repr.fine_joint_dict = None
    
    def contact_2d_general_flow(self, visualize=False):
        from embodied_analogy.utility.proposal.ram_proposal import get_ram_affordance_2d
        
        self.base_step()
        self.initial_frame = self.capture_frame()
        
        # 只在第一次进行 contact transfer, 之后直接进行复用
        print("Start transfering 2d contact affordance map...")
        self.affordance_map_2d = get_ram_affordance_2d(
            query_rgb=self.initial_frame.rgb,
            instruction=self.cfg["instruction"],
            obj_description=self.obj_description,
            fully_zeroshot=self.cfg["fully_zeroshot"],
            visualize=visualize
        )
        obj_mask = self.affordance_map_2d.get_obj_mask(visualize=False)
        contact_uv = self.affordance_map_2d.sample_highest(visualize=visualize)
        
        self.initial_frame.obj_mask = obj_mask
        self.initial_frame.contact2d = contact_uv
        
    def reconstruct_general_flow(self, visualize=False, gt_joint_type=False):
        """
        从初始 frame + 初始 contact point 得到的 flow 预测物体模型
        """
        # (M, T, 3) in camera frame
        general_flow = get_generalFlow(
            frame=self.initial_frame, 
            # TODO 这里设置为固定的 instruction
            instruction="open_Storage_Furniture", 
            visualize=visualize
        )
        # (M, T, 3) - > (T, M, 3)
        general_flow = np.transpose(general_flow, (1, 0, 2))
        self.general_flow_c = general_flow
        
        if gt_joint_type:
            joint_type = self.obj_repr.gt_joint_dict["joint_type"]
            if joint_type == "revolute":
                joint_dict, _ = coarse_R_from_tracks_3d(tracks_3d=general_flow, visualize=False)
                joint_dict["joint_type"] = "revolute"
            elif joint_type == "prismatic":
                joint_dict, _ = coarse_t_from_tracks_3d(tracks_3d=general_flow, visualize=False)
                joint_dict["joint_type"] = "prismatic"
        else:
            joint_dict = coarse_estimation(tracks_3d=general_flow, visualize=False)
        
        # 将 joint_dict 转换到世界坐标系下
        self.obj_repr.coarse_joint_dict = joint_dict
        self.obj_repr.fine_joint_dict = joint_dict
            
    def manip_general_flow(self, visualize=False):
        """
        根据 joint_dict 和 initial_frame 进行 manipulation
        """
        self.initial_frame.detect_grasp(
            use_anygrasp=self.cfg["use_anygrasp"],
            world_frame=True,
            visualize=visualize,
            asset_path=ASSET_PATH
        )
        # 
        pc_collision_w, pc_colors = self.initial_frame.get_env_pc(
            use_height_filter=False,
            world_frame=True
        )
        self.planner.update_point_cloud(pc_collision_w)
            
        if self.initial_frame.grasp_group is None or len(self.initial_frame.grasp_group) == 0: 
            return False
        
        for grasp_w in self.initial_frame.grasp_group:
            # 根据 best grasp 得到 pre_ph_grasp 和 ph_grasp 的位姿
            # 从 general_flow_c 中找到 dir_out_c
            first_frame_flow_c = self.general_flow_c[1, :, :] - self.general_flow_c[0, :, :] # M, 3
            # 由于已经运行了 detect_grasp, 所以 contact_3d 已经被设置过了
            nearest_flow = np.argmin(
                np.linalg.norm(self.initial_frame.contact3d - self.general_flow_c[0], axis=-1) # M
            ) # M
            # 由于我们产生 flow 的 prompt 固定是 open, 所以现在的 flow 天然的就是 out 
            dir_out_c = first_frame_flow_c[nearest_flow]
            dir_out_c = dir_out_c / max(np.linalg.norm(dir_out_c), 1e-8)
            dir_out_w = np.linalg.inv(self.initial_frame.Tw2c)[:3, :3] @ dir_out_c
            
            grasp = self.get_rotated_grasp(grasp_w, axis_out_w=dir_out_w)
            Tph2w = self.anyGrasp2ph(grasp=grasp)        
            Tph2w_pre = self.get_translated_ph(Tph2w, -self.cfg["reserved_distance"])
            result_pre = self.plan_path(target_pose=Tph2w_pre, wrt_world=True)
            
            if result_pre is not None:
                if visualize:
                    obj_pc, pc_colors = self.initial_frame.get_obj_pc(
                        use_robot_mask=True,
                        use_height_filter=True,
                        world_frame=True
                    )
                    Tc2w = np.linalg.inv(self.initial_frame.Tw2c)
                    contact3d_w = Tc2w[:3, :3] @ self.initial_frame.contact3d + Tc2w[:3, -1]
                    visualize_pc(
                        points=obj_pc, 
                        colors=pc_colors / 255,
                        grasp=grasp, 
                        contact_point=contact3d_w, 
                        post_contact_dirs=[dir_out_w]
                    )
                break
        
        # 没有成功的 planning 结果, 直接返回 False
        if result_pre is None:
            return False
        
        self.follow_path(result_pre)
        self.open_gripper()
        self.clear_planner_pc()
        self.move_forward(
            moving_distance=self.cfg["reserved_distance"],
            drop_large_move=False
        )
        self.close_gripper()
        
        joint_dict_w = self.obj_repr.get_joint_param(resolution="coarse", frame="world")
        self.move_along_axis(
            joint_type=joint_dict_w["joint_type"],
            joint_axis=joint_dict_w["joint_dir"],
            joint_start=joint_dict_w["joint_dir"],
            moving_distance=self.goal_delta,
            drop_large_move=False
        )
        
    def main_general_flow(self, visualize=False, gt_joint_type=False):
        self.contact_2d_general_flow(visualize=visualize)
        self.reconstruct_general_flow(visualize=False, gt_joint_type=gt_joint_type)
        self.manip_general_flow(visualize=visualize)
        
        manip_result = self.evaluate()
        return manip_result
    
    
if __name__ == "__main__":
    cfg = {
    "joint_type": "revolute",
    "data_path": "dataset/one_door_cabinet/46859_link_0",
    "obj_index": "46859",
    "joint_index": "0",
    "obj_description": "cabinet",
    "load_pose": [
        0.930869460105896,
        0.0,
        0.6496312618255615
    ],
    "load_quat": [
        0.9989797472953796,
        0.008722164668142796,
        -0.0003868725325446576,
        -0.04430985078215599
    ],
    "load_scale": 1,
    "active_link_name": "link_0",
    "active_joint_name": "joint_0",
    "instruction": "open the cabinet",
    "init_joint_state": 0.4801562746897319,
    "obj_folder_path_explore": "/home/zby/Programs/Embodied_Analogy/assets/logs/explore_512/46859_link_0",
    "phy_timestep": 0.004,
    "planner_timestep": 0.01,
    "use_sapien2": True,
    "fully_zeroshot": False,
    "record_fps": 30,
    "pertubation_distance": 0.1,
    "valid_thresh": 0.5,
    "max_tries": 10,
    "update_sigma": 0.05,
    "reserved_distance": 0.05,
    "num_initial_pts": 1000,
    "offscreen": True,
    "use_anygrasp": True,
    "obj_folder_path_reconstruct": "/home/zby/Programs/Embodied_Analogy/assets/logs/recon_512/46859_link_0",
    "num_kframes": 5,
    "fine_lr": 0.001,
    "save_memory": True,
    "scale_dir": "/home/zby/Programs/Embodied_Analogy/assets/logs/manip_512/46859_link_0/close/scale_10",
    "manipulate_type": "close",
    "manipulate_distance": 10.0,
    "reloc_lr": 0.003,
    "whole_traj_close_loop": True,
    "max_manip": 5.0,
    "prismatic_whole_traj_success_thresh": 0.01,
    "revolute_whole_traj_success_thresh": 5.0,
    "max_attempts": 5,
    "max_distance": 35.0,
    "prismatic_reloc_interval": 0.05,
    "prismatic_reloc_tolerance": 0.01,
    "revolute_reloc_interval": 5.0
}
    env = GeneralFlow_ManipEnv(cfg=cfg)
    print(env.main_general_flow(visualize=False, gt_joint_type=True))
    