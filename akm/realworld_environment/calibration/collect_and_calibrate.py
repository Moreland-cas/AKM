"""
对相机进行外参标定
"""
import os
import cv2
import yaml
import numpy as np
from PIL import Image
import time
from scipy.spatial.transform import Rotation as R
import shutil
from akm.realworld_environment.robot_env import RobotEnv
from akm.representation.basic_structure import Frame
from akm.utility.utils import visualize_pc


def sample_pos():
    # Change only one random translation axis
    rel_dir= np.random.uniform(-0.1, 0.1, (3, ))
    rel_dir = rel_dir / np.linalg.norm(rel_dir)
    rel_translation = rel_dir * np.random.uniform(0.0, 0.15)
    
    # Change only one random rotation axis (in degrees)
    # rel_rotation = np.random.uniform(-30, 30, (3, ))
    rel_rotation = np.random.uniform(-20, 20, (3, ))
    
    return rel_translation, rel_rotation
    
    
def collect_data(
    init_qpos=None,
    save_folder = "/home/user/Programs/Embodied_Analogy/assets/calib_data",
    num_views=30
):
    shutil.rmtree(save_folder)
    os.makedirs(save_folder, exist_ok=True)
    
    cfg_path = "/home/user/Programs/Embodied_Analogy/embodied_analogy/realworld_environment/calibration/test.yaml"

    # open
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    env = RobotEnv(cfg)
    env.franky_robot.recover_from_errors()

    # 控制机械臂走一个固定轨迹
    if init_qpos is not None:
        env.calibrate_reset(init_qpos)
    # input("Reset gripper to a good initial pose: ")
    """
    calibrate reset 对应的坐标系为:
    y(.) ---> x
    |
    v
    z
    所以最好调整 calibrate 的 init pose 到相机的中间
    """
    env.open_gripper(target=0.02)
    input("Make sure the checkerboard is in position before you type anything to close the gripper: ")
    env.close_gripper(gripper_force=60)

    # Tinit2w = env.robot.get_ee_pose(as_matrix=True)
    num_views_done = 0
    while num_views_done < num_views:
    # for i in range(num_views):
        # 当前要达到的位置
        try:
            tgt_t, tgt_r = sample_pos()
            
            # Ttgt2init = np.eye(4)
            # Ttgt2init[:3, -1] = tgt_t
            # Ttgt2init[:3, :3] = R.from_euler('xyz', tgt_r, degrees=True).as_matrix()
            # Ttgt2w = Tinit2w @ Ttgt2init
            # env.move_to_pose(Ttgt2w, wrt_world=True)
            
            # 实际去执行
            env.rot_dxyz(tgt_r)
            env.move_dxyz(tgt_t)
            
            # 等待一会儿再拍, 防止图片模糊
            time.sleep(1.)
            # 以下这个 frame 里已经包含了 rgb, depth 和 Tph2w
            f = env.capture_frame(robot_mask=False)
            # Image.fromarray(f.rgb).save(f'{save_folder}/{i}.png')
            
            # 同时获取当前的 Tph2w 并进行保存
            # Tph2w = env.robot.get_ee_pose(as_matrix=True)
            # np.save(f'{save_folder}/{i}.npy', Tph2w)
            f.save(file_path=os.path.join(save_folder, f'{num_views_done}.npy'))
            
            # 复位
            # Ttgt2w[:3, -1] -= tgt_t
            # env.move_to_pose(Ttgt2w, wrt_world=True)
            # env.move_to_pose(Tinit2w, wrt_world=True)
            env.move_dxyz(-tgt_t)
            inv_rot = R.from_euler('xyz', tgt_r, degrees=True).inv().as_euler('xyz', degrees=True)
            env.rot_dxyz(inv_rot)
            num_views_done += 1
        except Exception as e:
            print(f'Catched Exception: {str(e)}')
            env.franky_robot.recover_from_errors()
            env.calibrate_reset(init_qpos)
            
    print(f'Collect {num_views} images in {save_folder}.')
    
    
def estimate_intrinsic_extrinsic(calib_folder='/home/user/Programs/Embodied_Analogy/assets/calib_data/', visualize=False):
    frame_paths = []

    for f in os.listdir(calib_folder):
        if f.endswith(".npy"):
            frame_paths.append(os.path.join(calib_folder, f))
    print(f'Find {len(frame_paths)} frames.')

    #标定板的中长度对应的角点的个数
    XX = 11
    YY = 8
    #标定板一格的长度  单位为米
    L = 1.55 / 100.
    # L = 1.1 / 100.
    print(f'Using Checkerboard with (XX, YY) = ({XX}, {YY}) grids and L = {L * 100} cm.')

    obj_points = []     # 存储3D点
    img_points = []     # 存储2D点
    Tph2w_list = []
    for frame_path in frame_paths:
        frame = Frame.load(frame_path)
        
        img = frame.rgb
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        Tph2w = frame.Tph2w
        
        # 获取标定板角点的位置
        objp = np.zeros((XX * YY, 3), np.float32)
        # (2, 11, 8) -> (8, 11, 2) -> (88, 2)
        objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)     # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
        objp = L * objp

        size = gray.shape[::-1]  # (W, H)
        # ret: True/False, corners: (88, 1, 2)
        ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)
        
        # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
        
        if ret:
            obj_points.append(objp)
            # 在原角点的基础上寻找亚像素角点
            corners_refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  
            img_points.append(corners_refined)
            Tph2w_list.append(Tph2w)
            
        if visualize:
            cv2.drawChessboardCorners(img, (XX, YY), corners, ret)
            cv2.imshow('Corners', img)
            cv2.waitKey(0)
            
    print(f"Finished find_corners for all images, {len(obj_points)} of them are valid.")        

    # 标定,得到图案在相机坐标系下的位姿
    initial_mtx = np.array([
        [385.514, 0, 319.319],
        [0, 384.945, 244.607],
        [0, 0, 1]
    ])
    print(f'Initial intrinsic guess is {initial_mtx}.')
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS + 
        cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 +  # Fix radial distortion
        cv2.CALIB_FIX_TANGENT_DIST # Fix tangential distortion
    )  
    print("Assume estimation with initial guess and zero distortion.")
    distCoeffs = np.zeros((5, 1), dtype=np.float64)
    # 返回的 rvecs 和 tvecs 是 Tboard2camera, 且相机坐标系是 x 向右, y 向下, z 向前
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, initial_mtx, distCoeffs, flags=flags)
    print(f'Refined Intrinsic matrix: ')
    print(mtx)
    
    # Compute per-image reprojection error
    total_error = 0
    for i in range(len(obj_points)):
        img_points_projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i], img_points_projected, cv2.NORM_L2) / len(img_points_projected)
        print(f"Image {i+1} reprojection error: {error:.3f} pixels")
        total_error += error
    mean_error = total_error / len(obj_points)
    print(f"Mean per-image reprojection error: {mean_error:.3f} pixels")
    """
        void cv::calibrateHandEye	(	
        InputArrayOfArrays	R_gripper2base,
        InputArrayOfArrays	t_gripper2base,
        InputArrayOfArrays	R_target2cam,
        InputArrayOfArrays	t_target2cam,
        OutputArray	R_cam2gripper,
        OutputArray	t_cam2gripper,
        HandEyeCalibrationMethod	method = CALIB_HAND_EYE_TSAI )
        NOTE: 对于 hand wo camera 的情况, 我们要把 gripper 当 base, base 当 camera
    """
    Tbase2gripper_list = [np.linalg.inv(Tph2w) for Tph2w in Tph2w_list]
    R_base2gripper = [Tbase2gripper[:3, :3] for Tbase2gripper in Tbase2gripper_list]
    t_base2gripper = [Tbase2gripper[:3, -1] for Tbase2gripper in Tbase2gripper_list]

    # 返回的是 Tcamera2base (由于我们是eye to hand, 所以这里世实际返回的是 Tcamera2base 而不是 Tcamera2gripper)
    R, t = cv2.calibrateHandEye(R_base2gripper, t_base2gripper, rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI)
    Tcamera2w = np.eye(4)
    Tcamera2w[:3, :3] = R
    Tcamera2w[:3, -1] = t.reshape(-1)
    return mtx, Tcamera2w

"""
Find init qpos using: env.robot.get_qpos()
env.move_dx()
env.rot_dx()
Image.fromarray(env.capture_frame().rgb).show()
"""
if __name__ == "__main__":
    calib_folder = "/home/user/Programs/Embodied_Analogy/assets/calib_data"
    init_qpos=np.array([-2.55179204,  0.95217725,  2.49179559, -2.54909801, -0.31007478,
        1.89506812, -1.55903886,  0.        ,  0.        ])
    
    # collect_data(
    #     init_qpos=init_qpos,
    #     save_folder=calib_folder
    # )
    K, Tc2w = estimate_intrinsic_extrinsic(calib_folder=calib_folder)
    print("K:")
    print(K)
    print("Tc2w:")
    print(Tc2w)
    
    # 将 K 和 Tc2w 保存到当前文件夹下
    os.makedirs(os.path.join(calib_folder, 'tmp'), exist_ok=True)
    np.save(
        file=os.path.join(calib_folder, 'tmp/K.npy'),
        arr=K
    )
    np.save(
        file=os.path.join(calib_folder, 'tmp/Tc2w.npy'),
        arr=Tc2w
    )
    
    # 最后画一下在世界坐标系中的点云, 以验证世界坐标系估计的准确性
    f = Frame.load(os.path.join(calib_folder, "0.npy"))
    f.Tw2c = np.linalg.inv(Tc2w)
    f.K = K
    pc, colors =f.get_env_pc(
        use_robot_mask=False, 
        use_height_filter=True,
        world_frame=True,
        visualize=False
    )
    visualize_pc(
        points=pc,
        point_size=5,
        colors=colors/255.,
        online_viewer=True,
        visualize_origin=True
    )