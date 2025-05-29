import numpy as np
from torch.utils.data import Dataset, DataLoader
from opencood.visualization.vis_utils import visualize_sequence_dataloader
import glob
import os
import yaml

from opencood.utils.pcd_utils import load_lidar_bin
from opencood.utils.transformation_utils import x_to_world

lidar_folder = 'c:/Users/daehy/V2X-Real-main/data/test여분/2023-04-07-15-02-15_1_1/1'
vehicle_ID = 4  # None이면 전체 차량, 특정 ID를 지정하면 해당 차량만

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


def get_all_vehicle_bboxes(yaml_path, vehicle_id=None):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    vehicles = data.get('vehicles', {})
    bboxes = {}
    if vehicle_id is not None:
        vid_str = str(vehicle_id)
        if vid_str in vehicles:
            veh = vehicles[vid_str]
        elif isinstance(vehicle_id, int) and vehicle_id in vehicles:
            veh = vehicles[vehicle_id]
        else:
            raise ValueError(f"Vehicle ID {vehicle_id} not found in yaml.")
        x, y, z = veh['location']
        l, w, h = veh['extent']
        yaw = veh['angle'][2]
        bbox = np.array([x, y, z, 2*h, 2*w, 2*l, yaw], dtype=np.float32)
        bboxes[vid_str] = bbox
    else:
        for vid, veh in vehicles.items():
            x, y, z = veh['location']
            l, w, h = veh['extent']
            yaw = veh['angle'][2]
            bbox = np.array([x, y, z, 2*h, 2*w, 2*l, yaw], dtype=np.float32)
            bboxes[str(vid)] = bbox
    return bboxes  # {vehicle_id: bbox, ...}

class DummyDataset(Dataset):
    def __init__(self, folder, vehicle_id=None):
        self.bin_files = sorted(glob.glob(os.path.join(folder, '*.bin')))
        self.yaml_files = sorted(glob.glob(os.path.join(folder, '*.yaml')))
        self.vehicle_id = vehicle_id
        first_bboxes = get_all_vehicle_bboxes(self.yaml_files[0], vehicle_id)
        self.vehicle_ids = list(first_bboxes.keys())
        self.kf_dict = {}
        self.size_dict = {}
        for vid, bbox in first_bboxes.items():
            x, y, z, h, w, l, yaw = bbox
            self.kf_dict[vid] = KalmanFilter([x, y, z, yaw, 0, 0, 0, 0])
            self.size_dict[vid] = (h, w, l)

    def __len__(self):
        return len(self.bin_files)

    def __getitem__(self, idx):
        lidar = load_lidar_bin(self.bin_files[idx])
        with open(self.yaml_files[idx], 'r') as f:
            data = yaml.safe_load(f)
        pose = data.get('true_ego_pose', None)
        if pose is not None:
            lidar = transform_lidar_to_world(lidar, pose)
        bboxes = get_all_vehicle_bboxes(self.yaml_files[idx], self.vehicle_id)

        all_boxes = []
        all_masks = []
        for i, (vid, bbox) in enumerate(bboxes.items()):
            kf = self.kf_dict.get(vid)
            if kf is None:
                x, y, z, h, w, l, yaw = bbox
                kf = KalmanFilter([x, y, z, yaw, 0, 0, 0, 0])
                self.kf_dict[vid] = kf
                self.size_dict[vid] = (h, w, l)
            pred_state = kf.predict()
            obs = np.array([bbox[0], bbox[1], bbox[2], bbox[6]], dtype=np.float32)
            kf.update(obs)
            est_state = kf.x[:4]
            h, w, l = self.size_dict[vid]
            kf_box = np.array([est_state[0], est_state[1], est_state[2], h, w, l, est_state[3]], dtype=np.float32)
            all_boxes.append(bbox)
            all_masks.append(1)
            all_boxes.append(kf_box)
            all_masks.append(int(vid)+2)

        all_boxes = np.stack(all_boxes, axis=0)
        all_masks = np.array(all_masks, dtype=np.int32)

        return {
            'ego': {
                'origin_lidar': lidar,
                'object_bbx_center': all_boxes,
                'object_bbx_mask': all_masks
            }
        }

if __name__ == "__main__":
    # vehicle_ID를 지정하면 해당 차량만, None이면 전체 차량
    dummy_loader = DataLoader(DummyDataset(lidar_folder, vehicle_id=vehicle_ID), batch_size=1)
    visualize_sequence_dataloader(dummy_loader, order='hwl', color_mode='constant')