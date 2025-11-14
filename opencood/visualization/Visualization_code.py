import numpy as np
from torch.utils.data import Dataset, DataLoader
from opencood.visualization.vis_utils import visualize_sequence_dataloader, o3d
import glob
import os
import yaml
import random

from opencood.utils.pcd_utils import load_lidar_bin
from opencood.utils.transformation_utils import x_to_world

lidar_folder = 'd:/데이터파일/OPV2V/test/testoutput_CAV_data_2022-03-21-09-35-07_7/1'
vehicle_ID = None  # None이면 전체 차량, 특정 ID를 지정하면 해당 차량만

def transform_lidar_to_world(lidar, pose):
    # pose: [x, y, z, roll, yaw, pitch]
    T = x_to_world(pose)  # (4, 4)
    lidar_homo = np.concatenate([lidar[:, :3], np.ones((lidar.shape[0], 1))], axis=1)  # (N, 4)
    lidar_world = (T @ lidar_homo.T).T  # (N, 4)
    lidar_world = np.hstack([lidar_world[:, :3], lidar[:, 3:4]])  # (N, 4)
    return lidar_world

def iou_2d_from_bboxes(kf_bbox, gt_bbox):
    """
    2D IOU (BEV) 계산 — x, y, yaw, length, width 만 사용 (높이 무시)
    bboxes 형식: [x, y, z, h, w, l, yaw]
    반환: IOU (float, 0.0 ~ 1.0)
    """
    def rect_corners(bbox):
        x, y, _, h, w, l, yaw = bbox
        half_l = l / 2.0
        half_w = w / 2.0
        local = np.array([
            [-half_l, -half_w],
            [ half_l, -half_w],
            [ half_l,  half_w],
            [-half_l,  half_w]
        ], dtype=np.float64)
        c = np.cos(yaw); s = np.sin(yaw)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        return (local @ R.T) + np.array([x, y], dtype=np.float64)

    def polygon_area(poly):
        if poly is None or len(poly) < 3:
            return 0.0
        x = poly[:, 0]; y = poly[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def is_inside(p, a, b):
        # p가 a->b 엣지의 왼쪽(내부)인지 판정 (크로스프로덕트)
        return ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= 0

    def intersection_point(s, e, a, b):
        # 선분 s->e 와 a->b의 교점 (병렬이면 e 반환)
        x1, y1 = s; x2, y2 = e; x3, y3 = a; x4, y4 = b
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-9:
            return e.copy()
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
        return np.array([px, py], dtype=np.float64)

    def sutherland_hodgman(subject, clipper):
        # subject poly를 clipper poly로 클리핑 (둘 다 볼록 다각형)
        output = subject.tolist()
        for i in range(len(clipper)):
            input_list = output
            output = []
            if not input_list:
                break
            A = clipper[i]; B = clipper[(i + 1) % len(clipper)]
            S = np.array(input_list[-1])
            for E_pt in input_list:
                E = np.array(E_pt)
                if is_inside(E, A, B):
                    if not is_inside(S, A, B):
                        output.append(intersection_point(S, E, A, B))
                    output.append(E)
                elif is_inside(S, A, B):
                    output.append(intersection_point(S, E, A, B))
                S = E
        return np.array(output, dtype=np.float64)

    poly1 = rect_corners(kf_bbox)
    poly2 = rect_corners(gt_bbox)

    area1 = polygon_area(poly1)
    area2 = polygon_area(poly2)
    inter_poly = sutherland_hodgman(poly1, poly2)
    inter_area = polygon_area(inter_poly)

    union = area1 + area2 - inter_area
    if union <= 1e-9:
        return 0.0
    return float(inter_area / union)

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
        self.R = np.eye(4) * 1   # 관측 노이즈
        self.R[3, 3] = 5

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[:4]  # [x, y, z, yaw]

    def update(self, z):
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P


def get_all_vehicle_bboxes(yaml_path, vehicle_id=None, noise_std=0.05, big_noise_std=0.4, big_noise_prob=0.05):
    with open(yaml_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.UnsafeLoader) 
    vehicles = data.get('vehicles', {})
    bboxes = {}
    def add_noise(bbox):
        bbox[:3] += np.random.normal(0, noise_std, 3)
        bbox[6] += np.random.normal(0, noise_std)
        # 각 상태변수별로 big_noise_prob 확률로 큰 노이즈 추가
        for i in range(3):  # x, y, z
            if random.random() < big_noise_prob:
                bbox[i] += np.random.normal(0, big_noise_std)*2
        if random.random() < big_noise_prob:  # yaw
            bbox[6] += np.random.normal(0, big_noise_std)
        return bbox

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
        yaw = np.deg2rad(veh['angle'][1])
        bbox = np.array([x, y, z, 2*h, 2*w, 2*l, yaw], dtype=np.float32)
        bbox = add_noise(bbox)
        bboxes[vid_str] = bbox
    else:
        for vid, veh in vehicles.items():
            x, y, z = veh['location']
            l, w, h = veh['extent']
            yaw = np.deg2rad(veh['angle'][1])
            bbox = np.array([x, y, z, 2*h, 2*w, 2*l, yaw], dtype=np.float32)
            bbox = add_noise(bbox)
            bboxes[str(vid)] = bbox
    return bboxes  # {vehicle_id: bbox, ...}

# DummyDataset에서 mode 4, 5 관련 코드는 모두 삭제
class DummyDataset(Dataset):
    def __init__(self, folder, vehicle_id=None, mode=3, noise_std=0.05, big_noise_std=0.4, big_noise_prob=0.05):
        self.bin_files = sorted(glob.glob(os.path.join(folder, '*.bin')))
        self.pcd_files = sorted(glob.glob(os.path.join(folder, '*.pcd')))
        self.yaml_files = sorted(glob.glob(os.path.join(folder, '*.yaml')))
        self.vehicle_id = vehicle_id
        self.mode = mode  # 1: 정답만, 2: 칼만만, 3: 둘 다
        self.noise_std = noise_std
        self.big_noise_std = big_noise_std
        self.big_noise_prob = big_noise_prob
        self.files = self.bin_files + self.pcd_files
        self.files.sort()
        first_bboxes = get_all_vehicle_bboxes(
            self.yaml_files[0], vehicle_id, self.noise_std, self.big_noise_std, self.big_noise_prob)
        self.vehicle_ids = list(first_bboxes.keys())
        self.kf_dict = {}
        self.size_dict = {}
        for vid, bbox in first_bboxes.items():
            x, y, z, h, w, l, yaw = bbox
            self.kf_dict[vid] = KalmanFilter([x, y, z, yaw, 0, 0, 0, 0])
            self.size_dict[vid] = (h, w, l)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == '.bin':
            lidar = load_lidar_bin(file_path)
        elif ext == '.pcd':
            pcd = o3d.io.read_point_cloud(file_path)
            lidar = np.asarray(pcd.points)
            if len(lidar.shape) == 2 and lidar.shape[1] == 3:
                lidar = np.hstack([lidar, np.zeros((lidar.shape[0], 1))])
        else:
            raise ValueError(f"Unknown lidar file extension: {ext}")

        yaml_idx = idx if idx < len(self.yaml_files) else -1
        with open(self.yaml_files[yaml_idx], 'r') as f:
            data = yaml.load(f, Loader=yaml.UnsafeLoader)
        pose = data.get('true_ego_pose', None)
        if pose is not None:
            lidar = transform_lidar_to_world(lidar, pose)
        bboxes = get_all_vehicle_bboxes(
            self.yaml_files[yaml_idx], self.vehicle_id, self.noise_std, self.big_noise_std, self.big_noise_prob)

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

            if self.mode == 1:  # 정답만
                all_boxes.append(bbox)
                all_masks.append(1)
            elif self.mode == 2:  # 칼만만
                all_boxes.append(kf_box)
                all_masks.append(int(vid)+2)
            elif self.mode in [3, 4]:  # 둘 다 (GT + KF)
                all_boxes.append(bbox)
                all_masks.append(1)
                all_boxes.append(kf_box)
                all_masks.append(int(vid) + 2)

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
    # mode: 1=정답만, 2=칼만만, 3=둘 다, 4=IoU 출력 모드
    mode = 4
    dataset = DummyDataset(
        lidar_folder, vehicle_id=vehicle_ID, mode=mode,
        noise_std=0.05, big_noise_std=0.4, big_noise_prob=0.05
    )
    dummy_loader = DataLoader(dataset, batch_size=1)

    # mode 4에서는 시각화 대신 IoU 계산
    if mode == 3:
        print("=== IoU Evaluation Mode (no visualization) ===\n")
        iou_results = []

        for frame_idx, batch in enumerate(dummy_loader):
            ego_data = batch['ego']
            boxes = ego_data['object_bbx_center'][0].numpy()  # (N, 7)
            masks = ego_data['object_bbx_mask'][0].numpy()    # (N,)

            frame_ious = []
            # boxes에는 GT와 KF 박스가 순서대로 들어있음 (mode=3과 동일 구조)
            for i in range(0, len(boxes), 2):
                if i + 1 < len(boxes):
                    gt_box = boxes[i]
                    kf_box = boxes[i + 1]
                    iou = iou_2d_from_bboxes(kf_box, gt_box)
                    frame_ious.append(iou)
                    print(f"[Frame {frame_idx:03d}] Object {int(masks[i+1]-2):02d} IoU: {iou:.4f}")

            if frame_ious:
                avg_iou = np.mean(frame_ious)
                iou_results.append(avg_iou)
                print(f"  → Average IoU (frame {frame_idx}): {avg_iou:.4f}")

        if iou_results:
            print("\n=== Overall IoU Summary ===")
            print(f"Average IoU over {len(iou_results)} frames: {np.mean(iou_results):.4f}")

    else:
        # 기존 시각화 모드
        visualize_sequence_dataloader(dummy_loader, order='hwl', color_mode='constant')
