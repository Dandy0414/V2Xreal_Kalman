import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from opencood.visualization.vis_utils import visualize_sequence_dataloader
import glob
import os
import yaml

# bin 파일들이 들어있는 상위 폴더 경로
lidar_folder = 'data/test여분/2023-04-07-15-02-15_1_1/1'

def get_first_vehicle_bbox(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    # vehicles의 첫 번째 key를 찾음
    vehicles = data.get('vehicles', {})
    if not vehicles:
        raise ValueError("No vehicles found in yaml.")
    first_key = sorted(vehicles.keys(), key=lambda x: int(x))[0]
    veh = vehicles[first_key]
    # location: [x, y, z], extent: [l, w, h], angle: [roll, pitch, yaw]
    x, y, z = veh['location']
    l, w, h = veh['extent']  # 순서: l, w, h
    yaw = veh['angle'][2]    # roll, pitch, yaw
    # 바운딩 박스 포맷: [x, y, z, h, w, l, yaw]
    bbox = np.array([[x, y, z, h, w, l, yaw]], dtype=np.float32)
    return bbox

class DummyDataset(Dataset):
    def __init__(self, folder):
        # 폴더 내 모든 .bin 파일 경로를 정렬해서 리스트로 저장
        self.bin_files = sorted(glob.glob(os.path.join(folder, '*.bin')))
        self.yaml_files = sorted(glob.glob(os.path.join(folder, '*.yaml')))

    def __len__(self):
        return len(self.bin_files)

    def __getitem__(self, idx):
        lidar = np.fromfile(self.bin_files[idx], dtype=np.float32).reshape(-1, 4)
        bbx_center = get_first_vehicle_bbox(self.yaml_files[idx])
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