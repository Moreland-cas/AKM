import os
import cv2
import time
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R
from crisp_py.utils.geometry import Pose

from akm.realworld_envs.robot_env import RobotEnv
from akm.representation.basic_structure import Frame
from akm.utility.utils import visualize_pc


def sample_pos():
    """
    Sample random poses to capture an Image for callibration
    """
    rel_dir= np.random.uniform(-0.1, 0.1, (3, ))
    rel_dir = rel_dir / np.linalg.norm(rel_dir)
    rel_translation = rel_dir * np.random.uniform(0.03, 0.15)
    rel_rotation = np.random.uniform(-20, 20, (3, ))
    return rel_translation, rel_rotation
    

def get_tgtPose(curPose):
    tgt_t, tgt_r = sample_pos()
    cur_position = curPose.position
    cur_orientation = curPose.orientation
    tgt_position = cur_position + tgt_t
    delta_r = R.from_euler('xyz', tgt_r, degrees=True)
    tgt_orientation = delta_r * cur_orientation
    tgtPose = Pose(position=tgt_position, orientation=tgt_orientation)
    return tgtPose


def collect_data(
    init_Tph2w=None,
    save_folder = "/home/user/Programs/AKM/assets/calib_data",
    num_views=30
):
    """
    Collect Image data for callibration
    """
    shutil.rmtree(save_folder)
    os.makedirs(save_folder, exist_ok=True)
    
    cfg = {
        "exp_cfg": {
            "exp_folder": "/home/user/Programs/AKM/assets",
            "method_name": "ours",
            "save_cfg": False,
            "save_vis": False,
            "save_result": False,
            "realworld": True
        },
        "logging": {
            "level": "INFO"
        },
        "base_env_cfg": {
            "offscreen": True,
            "phy_timestep": 0.004,
            "use_sapien2": True,
            "planner_timestep": 0.1,
            "calib_folder": "/home/user/Programs/AKM/assets/calib_data"
        },
        "task_cfg": {
            "task_id": 0,
            "instruction": "open the cabinet"
        },
    }
    
    env = RobotEnv(cfg)
    env.open_gripper()
    # env.crisp_robot.home()
    speed = 0.03

    # Control the robotic arm to follow a fixed trajectory
    init_pose = Pose(position=init_Tph2w[:3, -1], orientation=R.from_matrix(init_Tph2w[:3, :3]))
    env.switch_mode("cartesian_impedance")
    env.crisp_robot.move_to(pose=init_pose, speed=speed)
    env.open_gripper(target=0.02)
    input("Make sure the checkerboard is in position before you type anything to close the gripper: ")
    env.close_gripper(gripper_force=60)

    num_views_done = 0
    while num_views_done < num_views:
        # try:
        # tgt_t, tgt_r = sample_pos()
        
        # env.rot_dxyz(tgt_r, speed=speed)
        # env.move_dxyz(tgt_t, speed=speed)
        env.crisp_robot.move_to(pose=get_tgtPose(curPose=init_pose), speed=0.05)
        
        # sleep for a while to avoid motion blur
        time.sleep(1.)
        f = env.capture_frame(robot_mask=False)
        f.save(file_path=os.path.join(save_folder, f'{num_views_done}.npy'))
        
        # Reset
        # env.move_dxyz(-tgt_t, speed=speed)
        # inv_rot = R.from_euler('xyz', tgt_r, degrees=True).inv().as_euler('xyz', degrees=True)
        # env.rot_dxyz(inv_rot, speed=speed)
        env.crisp_robot.move_to(pose=init_pose, speed=0.05)
        num_views_done += 1
        # except Exception as e:
        #     print(f'Catched Exception: {str(e)}')
        #     env.franky_robot.recover_from_errors()
        #     env.calibrate_reset(init_qpos)
            
    print(f'Collect {num_views} images in {save_folder}.')
    env.open_gripper()
    env.delete()
    
    
def estimate_intrinsic_extrinsic(calib_folder='/home/user/Programs/AKM/assets/calib_data/', visualize=False):
    frame_paths = []

    for f in os.listdir(calib_folder):
        if f.endswith(".npy"):
            frame_paths.append(os.path.join(calib_folder, f))
    print(f'Find {len(frame_paths)} frames.')

    # The number of corner points in the calibration plate
    XX = 11
    YY = 8
    # The length of one grid of the calibration plate is in meters
    L = 1.55 / 100.
    print(f'Using Checkerboard with (XX, YY) = ({XX}, {YY}) grids and L = {L * 100} cm.')

    obj_points = []     # Storing 3D points
    img_points = []     # Storing 2D points
    Tph2w_list = []
    for frame_path in frame_paths:
        frame = Frame.load(frame_path)
        
        img = frame.rgb
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        Tph2w = frame.Tph2w
        print("Tph2w: ", Tph2w)
        
        # Get the position of the corner points of the calibration plate
        objp = np.zeros((XX * YY, 3), np.float32)
        # (2, 11, 8) -> (8, 11, 2) -> (88, 2)
        # The world coordinate system is built on the calibration board. The Z coordinates of all points are 0, so only x and y need to be assigned.
        objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)     
        objp = L * objp

        size = gray.shape[::-1]  # (W, H)
        # ret: True/False, corners: (88, 1, 2)
        ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)
        
        # Set the parameters for finding sub-pixel corner points, and the stopping criteria used are a maximum number of cycles of 30 and a maximum error tolerance of 0.001
        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
        
        if ret:
            obj_points.append(objp)
            # Find sub-pixel corner points based on the original corner points
            corners_refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  
            img_points.append(corners_refined)
            Tph2w_list.append(Tph2w)
            
        if visualize:
            cv2.drawChessboardCorners(img, (XX, YY), corners, ret)
            cv2.imshow('Corners', img)
            cv2.waitKey(0)
            
    print(f"Finished find_corners for all images, {len(obj_points)} of them are valid.")        

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
    # The returned rvecs and tvecs are Tboard2camera, and the camera coordinate system is x right, y down, z forward
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
        NOTE: For handheld cameras, we should use the gripper as the base and the base as the camera.
    """
    Tbase2gripper_list = [np.linalg.inv(Tph2w) for Tph2w in Tph2w_list]
    R_base2gripper = [Tbase2gripper[:3, :3] for Tbase2gripper in Tbase2gripper_list]
    t_base2gripper = [Tbase2gripper[:3, -1] for Tbase2gripper in Tbase2gripper_list]

    # What is returned is Tcamera2base (because we are eye to hand, so what is actually returned here is Tcamera2base instead of Tcamera2gripper)
    R, t = cv2.calibrateHandEye(R_base2gripper, t_base2gripper, rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI)
    Tcamera2w = np.eye(4)
    Tcamera2w[:3, :3] = R
    Tcamera2w[:3, -1] = t.reshape(-1)
    return mtx, Tcamera2w

if __name__ == "__main__":
    calib_folder = "/home/user/Programs/AKM/assets/calib_data"
    collect= True
    estimate = True
    test_valid = True
    if collect:
        os.makedirs(calib_folder, exist_ok=True)
        init_Tph2w = np.eye(4)
        init_Tph2w[:3, -1] = [0.44, -0.26, 0.45]
        init_Tph2w[:3, :3] = np.array(
            [[ 0.60, -0.79,  0.11],
            [-0.80, -0.60,  0.04],
            [ 0.03, -0.11, -0.99]]
        )
        
        collect_data(
            init_Tph2w=init_Tph2w,
            save_folder=calib_folder
        )
    if estimate:
        K, Tc2w = estimate_intrinsic_extrinsic(calib_folder=calib_folder)
        print("K:")
        print(K)
        print("Tc2w:")
        print(Tc2w)
        
        os.makedirs(os.path.join(calib_folder, 'tmp'), exist_ok=True)
        np.save(
            file=os.path.join(calib_folder, 'tmp/K.npy'),
            arr=K
        )
        np.save(
            file=os.path.join(calib_folder, 'tmp/Tc2w.npy'),
            arr=Tc2w
        )
    
    # Finally, draw the point cloud in the world coordinate system to verify the accuracy of the world coordinate system estimation
    if test_valid:
        Tc2w = np.load(os.path.join(calib_folder, 'tmp/Tc2w.npy'))
        K = np.load(os.path.join(calib_folder, 'tmp/K.npy'))
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
            points=[pc],
            point_size=10,
            # colors=[colors/255.],
            online_viewer=True,
            visualize_origin=True,
        )