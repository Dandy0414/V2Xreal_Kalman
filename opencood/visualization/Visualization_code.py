import numpy as np
from torch.utils.data import Dataset, DataLoader
from opencood.visualization.vis_utils import visualize_sequence_dataloader
import glob
import os
import yaml

from opencood.utils.pcd_utils import load_lidar_bin
from opencood.utils.transformation_utils import x_to_world

lidar_folder = 'data/test여분/2023-04-07-15-02-15_1_1/1'
vehicle_ID = 4

def get_vehicle_bbox(yaml_path, vehicle_id):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    vehicles = data.get('vehicles', {})
    if not vehicles:
        raise ValueError("No vehicles found in yaml.")
    if str(vehicle_id) in vehicles:
        veh = vehicles[str(vehicle_id)]
    elif int(vehicle_id) in vehicles:
        veh = vehicles[int(vehicle_id)]
    else:
        raise ValueError(f"Vehicle ID {vehicle_id} not found in yaml.")
    x, y, z = veh['location']
    l, w, h = veh['extent']
    yaw = veh['angle'][2]
    bbox = np.array([[x, y, z, 2*h, 2*w, 2*l, yaw]], dtype=np.float32)
    return bbox


def transform_lidar_to_world(lidar, pose):
    # pose: [x, y, z, roll, yaw, pitch]
    T = x_to_world(pose)  # (4, 4)
    lidar_homo = np.concatenate([lidar[:, :3], np.ones((lidar.shape[0], 1))], axis=1)  # (N, 4)
    lidar_world = (T @ lidar_homo.T).T  # (N, 4)
    lidar_world = np.hstack([lidar_world[:, :3], lidar[:, 3:4]])  # (N, 4)
    return lidar_world


class KalmanFilter:
    def __init__(self, init_state, dt=0.1):
        # 상태: [x, y, z, yaw, vx, vy, vz, vyaw]
        self.dt = dt
        self.x = np.array(init_state, dtype=np.float32)  # 초기 상태
        self.P = np.eye(8) * 1.0  # 오차 공분산
        self.A = np.eye(8)
        for i in range(4):
            self.A[i, i+4] = dt  # 위치/방향에 속도/각속도 반영
        self.Q = np.eye(8) * 0.1  # 프로세스 노이즈
        self.H = np.eye(4, 8)      # 관측 행렬 (x, y, z, yaw만 관측)
        self.R = np.eye(4) * 0.1   # 관측 노이즈
        self.R[3, 3] = 30.0 

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[:4]  # [x, y, z, yaw]

    def update(self, z):
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P


class DummyDataset(Dataset):
    def __init__(self, folder):
        self.bin_files = sorted(glob.glob(os.path.join(folder, '*.bin')))
        self.yaml_files = sorted(glob.glob(os.path.join(folder, '*.yaml')))
        # 첫 번째 yaml 파일에서 초기값 추출
        with open(self.yaml_files[0], 'r') as f:
            data = yaml.safe_load(f)
        bbx_center = get_vehicle_bbox(self.yaml_files[0], vehicle_ID)
        # [x, y, z, yaw, vx, vy, vz, vyaw]로 초기화 (속도/각속도는 0으로)
        x, y, z = bbx_center[0, :3]
        yaw = bbx_center[0, 6]
        self.kf = KalmanFilter([x, y, z, yaw, 0, 0, 0, 0])

    def __len__(self):
        return len(self.bin_files)

    def __getitem__(self, idx):
        lidar = load_lidar_bin(self.bin_files[idx])
        with open(self.yaml_files[idx], 'r') as f:
            data = yaml.safe_load(f)
        pose = data.get('true_ego_pose', None)
        if pose is not None:
            lidar = transform_lidar_to_world(lidar, pose)
        bbx_center = get_vehicle_bbox(self.yaml_files[idx], vehicle_ID)
        bbx_mask = np.array([1], dtype=np.int32)

        # 1. 칼만필터 예측
        pred_state = self.kf.predict()  # [x, y, z, yaw]

        # 2. yaml에서 차량의 실제 위치+yaw(관측값) 읽기
        obs_xyz = bbx_center[0, :3]  # [x, y, z]
        obs_yaw = bbx_center[0, 6]   # yaw
        obs = np.array([obs_xyz[0], obs_xyz[1], obs_xyz[2], obs_yaw], dtype=np.float32)

        # 3. 칼만필터 업데이트(보정)
        self.kf.update(obs)

        # 4. 보정된 추정값으로 바운딩 박스 생성
        est_state = self.kf.x[:4]  # [x, y, z, yaw]
        h, w, l = 2, 2, 4
        kf_box = np.array([[est_state[0], est_state[1], est_state[2], h, w, l, est_state[3]]], dtype=np.float32)

        all_boxes = np.vstack([bbx_center, kf_box])
        all_masks = np.array([1, 2], dtype=np.int32)  # 실제: 1, 칼만: 2

        return {
            'ego': {
                'origin_lidar': lidar,
                'object_bbx_center': all_boxes,
                'object_bbx_mask': all_masks
            }
        }


if __name__ == "__main__":
    dummy_loader = DataLoader(DummyDataset(lidar_folder), batch_size=1)
    visualize_sequence_dataloader(dummy_loader, order='hwl', color_mode='constant')