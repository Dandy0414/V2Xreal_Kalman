import numpy as np
from torch.utils.data import Dataset, DataLoader
from opencood.visualization.vis_utils import visualize_sequence_dataloader
import glob
import os
import yaml

from opencood.utils.pcd_utils import load_lidar_bin
from opencood.utils.transformation_utils import x_to_world

lidar_folder = 'data/test여분/2023-04-07-15-02-15_1_1/1'


def get_vehicle_bbox(yaml_path, vehicle_id=4):
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
    bbox = np.array([[x, y, z, h, w, l, yaw]], dtype=np.float32)
    return bbox


def transform_lidar_to_world(lidar, pose):
    # pose: [x, y, z, roll, yaw, pitch]
    T = x_to_world(pose)  # (4, 4)
    lidar_homo = np.concatenate([lidar[:, :3], np.ones((lidar.shape[0], 1))], axis=1)  # (N, 4)
    lidar_world = (T @ lidar_homo.T).T  # (N, 4)
    lidar_world = np.hstack([lidar_world[:, :3], lidar[:, 3:4]])  # (N, 4)
    return lidar_world


class DummyDataset(Dataset):
    def __init__(self, folder):
        self.bin_files = sorted(glob.glob(os.path.join(folder, '*.bin')))
        self.yaml_files = sorted(glob.glob(os.path.join(folder, '*.yaml')))

    def __len__(self):
        return len(self.bin_files)

    def __getitem__(self, idx):
        lidar = load_lidar_bin(self.bin_files[idx])
        # yaml에서 true_ego_pose(혹은 lidar_pose) 읽기
        with open(self.yaml_files[idx], 'r') as f:
            data = yaml.safe_load(f)
        pose = data.get('true_ego_pose', None)
        if pose is not None:
            lidar = transform_lidar_to_world(lidar, pose)
        bbx_center = get_vehicle_bbox(self.yaml_files[idx])
        bbx_mask = np.array([1], dtype=np.int32)
        return {
            'ego': {
                'origin_lidar': lidar,
                'object_bbx_center': bbx_center,
                'object_bbx_mask': bbx_mask
            }
        }


if __name__ == "__main__":
    dummy_loader = DataLoader(DummyDataset(lidar_folder), batch_size=1)
    visualize_sequence_dataloader(dummy_loader, order='hwl', color_mode='constant')