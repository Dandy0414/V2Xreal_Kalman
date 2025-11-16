import numpy as np
from torch.utils.data import Dataset, DataLoader
from opencood.visualization.vis_utils import visualize_sequence_dataloader, o3d
import glob
import os
import yaml
import random
import pandas as pd

from opencood.utils.pcd_utils import load_lidar_bin
from opencood.utils.transformation_utils import x_to_world

# 경로 입력 (역슬래시 자동 변환)
lidar_folder = r'D:\데이터파일\OPV2V\test\testoutput_CAV_data_2022-03-17-11-02-23_2\1'
# 또는 os.path.normpath() 사용
lidar_folder = os.path.normpath(lidar_folder)

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

class ExtendedKalmanFilter:
    """
    EKF using bicycle (kinematic) model:
      x += v * cos(yaw) * dt
      y += v * sin(yaw) * dt
      yaw += (v / L) * tan(delta) * dt
    State: [x, y, yaw, v, delta]
    Measurements: [x, y, yaw]
    """
    def __init__(self, init_state, dt=0.1, L=2.5):
        # init_state must be length 5: [x, y, yaw, v, delta]
        self.dt = dt
        self.L = L
        self.x = np.array(init_state, dtype=np.float64)
        # normalize yaw
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))
        self.P = np.eye(5) * 1.0
        self.Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.1])  # tune as needed
        self.R = np.diag([1, 1, 1.0])             # measurement noise

    def _normalize_angle(self, a):
        return np.arctan2(np.sin(a), np.cos(a))

    def _f(self, x):
        x_new = x.copy()
        yaw = x[2]; v = x[3]; delta = x[4]
        dt = self.dt; L = self.L
        x_new[0] = x[0] + v * np.cos(yaw) * dt
        x_new[1] = x[1] + v * np.sin(yaw) * dt
        x_new[2] = x[2] + (v / L) * np.tan(delta) * dt
        x_new[2] = self._normalize_angle(x_new[2])
        # keep v, delta (could add simple dynamics if desired)
        return x_new

    def _F_jacobian(self, x):
        yaw = x[2]; v = x[3]; delta = x[4]
        dt = self.dt; L = self.L
        F = np.eye(5, dtype=np.float64)
        # partials for x
        F[0,2] = -v * np.sin(yaw) * dt
        F[0,3] =  np.cos(yaw) * dt
        # partials for y
        F[1,2] =  v * np.cos(yaw) * dt
        F[1,3] =  np.sin(yaw) * dt
        # partials for yaw
        F[2,3] = (1.0 / L) * np.tan(delta) * dt
        F[2,4] = (v / L) * (1.0 / (np.cos(delta) ** 2)) * dt
        return F

    def _h(self, x):
        # measurement function: observe x, y, yaw
        return np.array([x[0], x[1], x[2]], dtype=np.float64)

    def _H_jacobian(self, x):
        H = np.zeros((3, 5), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        return H

    def predict(self):
        # predict state
        self.x = self._f(self.x)
        F = self._F_jacobian(self.x)
        self.P = F @ self.P @ F.T + self.Q
        return self._h(self.x)

    def update(self, z):
        # z: [x, y, yaw]
        z = np.array(z, dtype=np.float64)
        z_pred = self._h(self.x)
        y = z - z_pred
        # normalize yaw residual
        y[2] = self._normalize_angle(y[2])
        H = self._H_jacobian(self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = self._normalize_angle(self.x[2])
        self.P = (np.eye(5) - K @ H) @ self.P


class KalmanFilter:
    def __init__(self, init_state, dt=0.1):
        # 상태: [x, y, z, yaw, vx, vy, vz, vyaw]
        self.dt = dt
        self.x = np.array(init_state, dtype=np.float32)  # 초기 상태
        # normalize yaw 초기값
        self.x[3] = np.arctan2(np.sin(self.x[3]), np.cos(self.x[3]))
        self.P = np.eye(8) * 1.0  # 오차 공분산
        self.A = np.eye(8)
        for i in range(4):
            self.A[i, i+4] = dt  # 위치/방향에 속도/각속도 반영
        self.Q = np.eye(8) * 0.1 # 프로세스 노이즈
        self.H = np.eye(4, 8)      # 관측 행렬 (x, y, z, yaw만 관측)
        self.R = np.eye(4) * 1   # 관측 노이즈
        self.R[3, 3] = 5

    def predict(self):
        self.x = self.A @ self.x
        # yaw 정규화 (예측 후)
        self.x[3] = np.arctan2(np.sin(self.x[3]), np.cos(self.x[3]))
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[:4]  # [x, y, z, yaw]

    def update(self, z):
        # 선형 잔차 계산
        y = z - (self.H @ self.x)
        # yaw 잔차 정규화 -> 큰 꺾임 방지
        y[3] = np.arctan2(np.sin(y[3]), np.cos(y[3]))
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        # 업데이트 후 yaw 정규화
        self.x[3] = np.arctan2(np.sin(self.x[3]), np.cos(self.x[3]))
        self.P = (np.eye(8) - K @ self.H) @ self.P


def get_all_vehicle_bboxes(yaml_path, vehicle_id=None, noise_std=0.05, big_noise_std=0.4, big_noise_prob=0.05, add_noise_flag=True):
    """
    add_noise_flag=True: 노이즈 추가 (관측값 시뮬레이션)
    add_noise_flag=False: 순수 정답지 (노이즈 없음)
    """
    with open(yaml_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.UnsafeLoader) 
    vehicles = data.get('vehicles', {})
    bboxes = {}
    
    def add_noise(bbox):
        if not add_noise_flag:  # 노이즈 추가 안 함
            return bbox
        bbox[:3] += np.random.normal(0, noise_std, 3)
        bbox[6] += np.random.normal(0, noise_std)
        for i in range(3):
            if random.random() < big_noise_prob:
                bbox[i] += np.random.normal(0, big_noise_std)*2
        if random.random() < big_noise_prob:
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
        if add_noise_flag:
            bbox = add_noise(bbox)
        bboxes[vid_str] = bbox
    else:
        for vid, veh in vehicles.items():
            x, y, z = veh['location']
            l, w, h = veh['extent']
            yaw = np.deg2rad(veh['angle'][1])
            bbox = np.array([x, y, z, 2*h, 2*w, 2*l, yaw], dtype=np.float32)
            if add_noise_flag:
                bbox = add_noise(bbox)
            bboxes[str(vid)] = bbox
    return bboxes  # {vehicle_id: bbox, ...}

# DummyDataset에서 mode 4, 5 관련 코드는 모두 삭제
class DummyDataset(Dataset):
    def __init__(self, folder, vehicle_id=None, mode=3, noise_std=0.05, big_noise_std=0.4, big_noise_prob=0.05, use_obs_as_gt=False, use_ekf=False):
        self.bin_files = sorted(glob.glob(os.path.join(folder, '*.bin')))
        self.pcd_files = sorted(glob.glob(os.path.join(folder, '*.pcd')))
        self.yaml_files = sorted(glob.glob(os.path.join(folder, '*.yaml')))
        self.vehicle_id = vehicle_id
        self.mode = mode  # 1: 정답만, 2: 칼만만, 3: 둘 다
        self.use_obs_as_gt = use_obs_as_gt  # mode==3일 때 GT 대신 관측값 사용 여부
        self.use_ekf = use_ekf  # EKF 사용 여부
        self.noise_std = noise_std
        self.big_noise_std = big_noise_std
        self.big_noise_prob = big_noise_prob
        self.files = self.bin_files + self.pcd_files
        self.files.sort()
        
        # 정답지 (노이즈 없음)
        first_bboxes_gt = get_all_vehicle_bboxes(
            self.yaml_files[0], vehicle_id, self.noise_std, self.big_noise_std, self.big_noise_prob, add_noise_flag=False)
        
        # 관측값 (노이즈 있음) - 칼만필터 초기화용
        first_bboxes_obs = get_all_vehicle_bboxes(
            self.yaml_files[0], vehicle_id, self.noise_std, self.big_noise_std, self.big_noise_prob, add_noise_flag=True)
        
        self.vehicle_ids = list(first_bboxes_gt.keys())
        self.kf_dict = {}
        self.size_dict = {}
        for vid, bbox_gt in first_bboxes_gt.items():
            bbox_obs = first_bboxes_obs[vid]
            x, y, z, h, w, l, yaw = bbox_obs
            
            if self.use_ekf:
                # EKF 초기화 (bicycle model state: [x, y, yaw, v, delta])
                self.kf_dict[vid] = ExtendedKalmanFilter([x, y, yaw, 0.0, 0.0])
            else:
                # 선형 칼만필터 초기화
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
        
        # 정답지 (노이즈 없음)
        bboxes_gt = get_all_vehicle_bboxes(
            self.yaml_files[yaml_idx], self.vehicle_id, self.noise_std, self.big_noise_std, self.big_noise_prob, add_noise_flag=False)
        
        # 관측값 (노이즈 있음)
        bboxes_obs = get_all_vehicle_bboxes(
            self.yaml_files[yaml_idx], self.vehicle_id, self.noise_std, self.big_noise_std, self.big_noise_prob, add_noise_flag=True)

        all_boxes = []
        all_masks = []
        for vid, bbox_gt in bboxes_gt.items():
            bbox_obs = bboxes_obs[vid]
            kf = self.kf_dict.get(vid)
            if kf is None:
                x, y, z, h, w, l, yaw = bbox_obs
                if self.use_ekf:
                    # EKF expects [x, y, yaw, v, delta]
                    kf = ExtendedKalmanFilter([x, y, yaw, 0.0, 0.0])
                else:
                    kf = KalmanFilter([x, y, z, yaw, 0, 0, 0, 0])
                self.kf_dict[vid] = kf
                self.size_dict[vid] = (h, w, l)
            
            pred_state = kf.predict()
            
            if self.use_ekf:
                # EKF: 관측값 (x, y, yaw)
                obs = np.array([bbox_obs[0], bbox_obs[1], bbox_obs[6]], dtype=np.float32)
                kf.update(obs)
                est_state = kf.x[:3]  # [x, y, yaw]
                h, w, l = self.size_dict[vid]
                kf_box = np.array([est_state[0], est_state[1], bbox_obs[2], h, w, l, est_state[2]], dtype=np.float32)
            else:
                # 선형 칼만필터: 관측값 (x, y, z, yaw)
                obs = np.array([bbox_obs[0], bbox_obs[1], bbox_obs[2], bbox_obs[6]], dtype=np.float32)
                kf.update(obs)
                est_state = kf.x[:4]
                h, w, l = self.size_dict[vid]
                kf_box = np.array([est_state[0], est_state[1], est_state[2], h, w, l, est_state[3]], dtype=np.float32)

            if self.mode == 1:
                all_boxes.append(bbox_gt)  # 정답지만
                all_masks.append(1)
            elif self.mode == 2:
                all_boxes.append(kf_box)  # 칼만만
                all_masks.append(int(vid)+2)
            elif self.mode in [3, 4]:
                # 간단 처리: use_obs_as_gt=True이면 '관측값'을 정답지(마스크=1)처럼 표시
                if self.use_obs_as_gt:
                    all_boxes.append(bbox_obs)   # 관측값 (노이즈 있음)
                    all_masks.append(1)          # 정답지와 동일한 마스크(초록색으로 표시되는 기존 동작 재사용)
                else:
                    all_boxes.append(bbox_gt)    # 정답지 (노이즈 없음)
                    all_masks.append(1)
                
                all_boxes.append(kf_box)    # 칼만/EKF 필터 결과
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
    
    # use_obs_as_gt=True일 때: 정답지 대신 관측값(노이즈 있음) 표시
    # use_obs_as_gt=False일 때: 정답지(노이즈 없음) 표시
    use_obs_as_gt = True
    
    # EKF 사용 여부 (True: EKF 사용, False: 선형 칼만필터 사용)
    use_ekf = True
    
    dataset = DummyDataset(
        lidar_folder, vehicle_id=vehicle_ID, mode=mode,
        noise_std=0.05, big_noise_std=0.2, big_noise_prob=0.01,
        use_obs_as_gt=use_obs_as_gt,
        use_ekf=use_ekf  # ← EKF 옵션
    )
    
    # 프레임 수 확인
    num_frames = len(dataset)
    print(f"Total frames in dataset: {num_frames}\n")
    
    # DataLoader 설정 (shuffle=False, batch_size=1)
    dummy_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # mode 4에서는 시각화 대신 IoU 계산
    if mode == 4:
        print("=== IoU Evaluation Mode (no visualization) ===\n")
        detailed_results = []  # 프레임별 상세 정보
        vehicle_ious = {}  # 차량별 IOU 수집용

        import time
        start_time = time.time()

        for frame_idx, batch in enumerate(dummy_loader):
            # 진행도 바 출력
            progress = (frame_idx + 1) / num_frames
            bar_length = 40
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            percent = progress * 100
            
            ego_data = batch['ego']
            boxes = ego_data['object_bbx_center'][0].numpy()  # (N, 7)
            masks = ego_data['object_bbx_mask'][0].numpy()    # (N,)

            # boxes에는 GT와 KF 박스가 순서대로 들어있음 (mode=3과 동일 구조)
            for i in range(0, len(boxes), 2):
                if i + 1 < len(boxes):
                    gt_box = boxes[i]
                    kf_box = boxes[i + 1]
                    iou = iou_2d_from_bboxes(kf_box, gt_box)
                    
                    vehicle_id = int(masks[i+1] - 2)
                    
                    # 차량별 IOU 저장
                    if vehicle_id not in vehicle_ious:
                        vehicle_ious[vehicle_id] = {}
                    vehicle_ious[vehicle_id][frame_idx + 1] = iou
                    
                    # 상세 정보 저장 (ID, frame 각각 저장)
                    detailed_results.append({
                        'vehicle_id': vehicle_id,
                        'frame': frame_idx + 1,
                        'iou': iou
                    })
            
            # 진행도 바 출력
            progress = (frame_idx + 1) / num_frames
            bar_length = 40
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            percent = progress * 100
            print(f"\r[{bar}] {percent:.1f}% ({frame_idx + 1}/{num_frames})", end="", flush=True)

        elapsed_total = time.time() - start_time
        print("\n")  # 줄 바꿈
        
        if detailed_results:
            # 화면 출력 - 전체 통계
            print("\n=== Overall IoU Summary ===")
            all_ious = [item['iou'] for item in detailed_results]
            print(f"Total samples processed: {len(all_ious)}")
            print(f"Average IoU (all samples): {np.mean(all_ious):.4f}")
            print(f"Min IoU: {np.min(all_ious):.4f}")
            print(f"Max IoU: {np.max(all_ious):.4f}")
            print(f"Std IoU: {np.std(all_ious):.4f}")
            print(f"Processing time: {elapsed_total:.2f}s")

            # 화면 출력 - 차량별 통계
            print("\n=== Per-Vehicle IoU Summary ===")
            vehicle_summary = []
            for vehicle_id in sorted(vehicle_ious.keys()):
                ious = np.array(list(vehicle_ious[vehicle_id].values()))
                mean_iou = np.mean(ious)
                std_iou = np.std(ious)
                min_iou = np.min(ious)
                max_iou = np.max(ious)
                
                print(f"Vehicle {vehicle_id:02d}: frames={len(ious)}, mean={mean_iou:.4f}, std={std_iou:.4f}, min={min_iou:.4f}, max={max_iou:.4f}")
                
                vehicle_summary.append({
                    'vehicle_id': vehicle_id,
                    'frame_count': len(ious),
                    'mean_iou': mean_iou,
                    'std_iou': std_iou,
                    'min_iou': min_iou,
                    'max_iou': max_iou
                })

            # 차량 1번의 평균 IOU 출력
            if 1 in vehicle_ious:
                vehicle_1_ious = np.array(list(vehicle_ious[1].values()))
                vehicle_1_mean_iou = np.mean(vehicle_1_ious)
                print(f"\n*** Vehicle 1 Mean IOU: {vehicle_1_mean_iou:.4f} ***")
            else:
                print("\n*** Vehicle 1 not found in dataset ***")

            # CSV 저장 (프레임별 상세 정보 - vehicle_id와 frame 각각 저장)
            df_detailed = pd.DataFrame(detailed_results)
            df_detailed = df_detailed[['vehicle_id', 'frame', 'iou']]  # 컬럼 순서 재정렬
            df_detailed.to_csv("iou_detailed_results.csv", index=False)

            # CSV 저장 (2D 형식 - 가로: frame, 세로: vehicle_id)
            # vehicle_id를 인덱스로, frame을 컬럼으로 하는 피벗 테이블
            df_pivot = df_detailed.pivot(index='vehicle_id', columns='frame', values='iou')
            df_pivot.to_csv("iou_2d_results.csv")

            # CSV 저장 (차량별 평균 IOU)
            df_vehicle_summary = pd.DataFrame(vehicle_summary)
            df_vehicle_summary.to_csv("iou_vehicle_summary.csv", index=False)

            print("\n✓ CSV 저장 완료:")
            print("  - iou_detailed_results.csv (vehicle_id | frame | iou)")
            print("  - iou_2d_results.csv (2D 형식: 세로=vehicle_id, 가로=frame)")
            print("  - iou_vehicle_summary.csv (차량별 통계)")

    else:
        # 기존 시각화 모드
        visualize_sequence_dataloader(dummy_loader, order='hwl', color_mode='constant')
