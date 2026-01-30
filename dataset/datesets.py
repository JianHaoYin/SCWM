import numpy as np
import torch
import os
from PIL import Image
from typing import Tuple
import yaml
import pickle
import tqdm
from torch.utils.data import Dataset
from misc import angle_difference, get_data_path, get_delta_np, normalize_data, to_local_coords
import json
import torch.nn.functional as F

from pathlib import Path
ROOT = Path(__file__).resolve().parent

class BaseDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int, 
        context_size: int,
        transform: object,
        traj_names: str,
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        self.goals_per_obs = goals_per_obs


        traj_names_file = os.path.join(data_split_folder, traj_names)
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.image_size = image_size
        self.distance_categories = list(range(min_dist_cat, max_dist_cat + 1))
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.len_traj_pred = len_traj_pred
        self.traj_stride = traj_stride

        self.context_size = context_size
        self.normalize = normalize

        # load data/data_config.yaml
        with open("config/data_config.yaml", "r") as f:
            all_data_config = yaml.safe_load(f)

        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        # use this index to retrieve the dataset name from the data_config.yaml
        self.data_config = all_data_config[self.dataset_name]
        self.transform = transform
        self._load_index(predefined_index)
        self.ACTION_STATS = {}
        for key in all_data_config['action_stats']:
            self.ACTION_STATS[key] = np.expand_dims(all_data_config['action_stats'][key], axis=0)

    def _load_index(self, predefined_index) -> None:
        """
        Generates a list of tuples of (obs_traj_name, goal_traj_name, obs_time, goal_time) for each observation in the dataset
        """
        if predefined_index:
            print(f"****** Using a predefined evaluation index... {predefined_index}******")
            with open(predefined_index, "rb") as f:
                self.index_to_data = pickle.load(f)
                return
        else:
            print("****** Evaluating from NON PREDEFINED index... ******")
            index_to_data_path = os.path.join(
                self.data_split_folder,
                f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_n{self.context_size}_len_traj_pred_{self.len_traj_pred}.pkl",
            )
            
            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _build_index(self, use_tqdm: bool = False):
        """
        Build an index consisting of tuples (trajectory name, time, max goal distance)
        """
        samples_index = []
        goals_index = []

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])
            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size - 1
            end_time = traj_len - self.len_traj_pred
            for curr_time in range(begin_time, end_time, self.traj_stride):
                max_goal_distance = min(self.max_dist_cat, traj_len - curr_time - 1)
                min_goal_distance = max(self.min_dist_cat, -curr_time)
                samples_index.append((traj_name, curr_time, min_goal_distance, max_goal_distance))

        return samples_index, goals_index
  
    def _get_trajectory(self, trajectory_name):
        with open(os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
            traj_data = pickle.load(f)
        for k,v in traj_data.items():
            traj_data[k] = v.astype('float')
        return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data)

    def _compute_actions(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred + 1
        yaw = traj_data["yaw"][start_index:end_index]
        positions = traj_data["position"][start_index:end_index]
        goal_pos = traj_data["position"][goal_time]
        goal_yaw = traj_data["yaw"][goal_time]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            raise ValueError("is used?")
            # const_len = self.len_traj_pred + 1 - yaw.shape[0]
            # yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            # positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

        waypoints_pos = to_local_coords(positions, positions[0], yaw[0])
        waypoints_yaw = angle_difference(yaw[0], yaw)
        actions = np.concatenate([waypoints_pos, waypoints_yaw.reshape(-1, 1)], axis=-1)
        actions = actions[1:]
        
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])
        goal_yaw = angle_difference(yaw[0], goal_yaw)
        
        if self.normalize:
            actions[:, :2] /= self.data_config["metric_waypoint_spacing"]
            goal_pos[:, :2] /= self.data_config["metric_waypoint_spacing"]
        
        goal_pos = np.concatenate([goal_pos, goal_yaw.reshape(-1, 1)], axis=-1)
        return actions, goal_pos    
    def _get_meta(self, trajectory_name: str) -> dict:
        """
        Load meta.json for a trajectory folder.

        Expected structure:
            meta["records"][t]["camera"]["K"]    -> 3x3
            meta["records"][t]["camera"]["T_wc"] -> 4x4 (cam-to-world)
        """
        meta_path = os.path.join(self.data_folder, trajectory_name, "metadata.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"meta.json not found: {meta_path}")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return meta
    def _validate_meta_for_traj(self, trajectory_name: str) -> bool:
        """
        轨迹级别 metadata 校验：用于 build index 时过滤掉坏轨迹。
        要求：
        - metadata.json 存在且可读
        - records 是非空 list
        - 至少第 0 帧含 camera.K / camera.T_wc 且形状正确
        """
        try:
            meta = self._get_meta(trajectory_name)
        except Exception:
            return False

        records = meta.get("records", None)
        if not isinstance(records, list) or len(records) == 0:
            return False

        rec0 = records[0]
        if not isinstance(rec0, dict):
            return False

        cam = rec0.get("camera", None)
        if not isinstance(cam, dict):
            return False

        K = cam.get("K", None)
        T_wc = cam.get("T_wc", None)
        if K is None or T_wc is None:
            return False

        # 形状粗检（避免奇怪的 list）
        try:
            if not (len(K) == 3 and len(K[0]) == 3 and len(K[1]) == 3 and len(K[2]) == 3):
                return False
            if not (len(T_wc) == 4 and len(T_wc[0]) == 4 and len(T_wc[1]) == 4 and len(T_wc[2]) == 4 and len(T_wc[3]) == 4):
                return False
        except Exception:
            return False

        return True
    def _get_camera_params(self, meta: dict, t: int):
        """
        Return (K, T_wc) as torch.float32 tensors.
        K:   [3,3]
        T_wc:[4,4]
        """


        records = meta["records"]
        n = len(records)
        if t < 0 or t >= n:
            # 尽量打印能唯一定位样本的信息
            print("[BAD SAMPLE] t=", t, "len(records)=", n,
                "meta keys=", list(meta.keys()))
            # 如果 meta 里有路径 / id / scene / timestamp，一并打印
            for k in ["scene", "seq", "sequence", "id", "uid", "path", "folder", "name"]:
                if k in meta:
                    print(f"  meta[{k}] =", meta[k])
            # 打印前几个 record 的可辨识字段
            if n > 0 and isinstance(records[0], dict):
                print("  record0 keys:", list(records[0].keys()))
                for kk in ["t", "frame_id", "timestamp", "image", "img_path"]:
                    if kk in records[0]:
                        print("  record0", kk, "=", records[0][kk])
            raise IndexError(f"t out of range: t={t}, n={n}")

        rec = meta["records"][t]
        K = torch.as_tensor(rec["camera"]["K"], dtype=torch.float32)
        T_wc = torch.as_tensor(rec["camera"]["T_wc"], dtype=torch.float32)
        return K, T_wc

    def _get_satellite_image(self, trajectory_name: str) -> torch.Tensor:
        """
        Load a satellite image associated with this trajectory.

        You MUST decide your actual satellite image naming.
        Here we try a few common filenames; if none exist, we fallback to the first frame.
        Return: [3,H,W] float tensor AFTER self.transform.
        """
        cand = [
            "satellite.png", "satellite.jpg", "sat.png", "sat.jpg",
            "map.png", "map.jpg", "overhead.png", "overhead.jpg"
        ]
        folder = os.path.join(self.data_folder, trajectory_name)
        sat_path = None
        for name in cand:
            p = os.path.join(folder, name)
            if os.path.isfile(p):
                sat_path = p
                break

        if sat_path is None:
            # Fallback: use timestep 0 frame as a placeholder (NOT ideal, but keeps pipeline running)
            sat_path = get_data_path(self.data_folder, trajectory_name, 0)

        sat_img = self.transform(Image.open(sat_path))  # [3,H,W]
        return sat_img

class TrainingDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int, 
        context_size: int,
        transform: object,
        traj_names: str = 'traj_names.txt',
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        super().__init__(data_folder, data_split_folder, dataset_name, image_size, min_dist_cat, max_dist_cat,
            len_traj_pred, traj_stride, context_size, transform, traj_names, normalize, predefined_index, goals_per_obs)


    def __getitem__(self, i: int):
        """
        Returns:
          obs_image:   [context_size + goals_per_obs, 3, H, W]
          goal_pos:    [goals_per_obs, 3]
          rel_time:    [goals_per_obs]
          sat_img:     [3, Hs, Ws]   (per-trajectory satellite image)
          cam2world:   [goals_per_obs, 4, 4]  (T_wc for each goal frame)
          intrinsics:  [goals_per_obs, 3, 3]  (K for each goal frame)
        """
        try:
            f_curr, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[i]

            # Sample goals
            goal_offset = np.random.randint(min_goal_dist, max_goal_dist + 1, size=(self.goals_per_obs))
            goal_time = (curr_time + goal_offset).astype("int")
            rel_time = (goal_offset).astype("float") / 128.0  # keep your original normalization

            # Context frames + goal frames (for x pixels)
            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
            context = [(f_curr, t) for t in context_times] + [(f_curr, int(t)) for t in goal_time]

            obs_image = torch.stack(
                [self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context],
                dim=0,
            )  # [context_size + goals_per_obs, 3, H, W]

            # Trajectory data for actions
            curr_traj_data = self._get_trajectory(f_curr)

            # Compute actions/goal positions
            _, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)
            goal_pos[:, :2] = normalize_data(goal_pos[:, :2], self.ACTION_STATS)

            # --- NEW: satellite image (per trajectory) ---
            sat_img = self._get_satellite_image(f_curr)  # [3,Hs,Ws]

            # --- NEW: camera params for each goal frame (T_wc & K) ---
            meta = self._get_meta(f_curr)

            Ks = []
            T_wcs = []
            for gt in goal_time.tolist():
                K, T_wc = self._get_camera_params(meta, int(gt))
                Ks.append(K)
                T_wcs.append(T_wc)

            intrinsics = torch.stack(Ks, dim=0)  # [goals_per_obs,3,3]
            cam2world = torch.stack(T_wcs, dim=0)  # [goals_per_obs,4,4]

            return (
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(goal_pos, dtype=torch.float32),
                torch.as_tensor(rel_time, dtype=torch.float32),
                torch.as_tensor(sat_img, dtype=torch.float32),
                cam2world,
                intrinsics,
            )

        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise

class EvalDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int, 
        context_size: int,
        transform: object,
        traj_names: str,
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        super().__init__(data_folder, data_split_folder, dataset_name, image_size, min_dist_cat, max_dist_cat,
            len_traj_pred, traj_stride, context_size, transform, traj_names, normalize, predefined_index, goals_per_obs)
  
    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            f_curr, curr_time, _, _ = self.index_to_data[i]
            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
            pred_times = list(range(curr_time + 1, curr_time + self.len_traj_pred + 1))
            
            context = [(f_curr, t) for t in context_times]
            pred = [(f_curr, t) for t in pred_times]

            obs_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])
            pred_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in pred])

            curr_traj_data = self._get_trajectory(f_curr)

            # Compute actions
            actions, _ = self._compute_actions(curr_traj_data, curr_time, np.array([curr_time+1])) # last argument is dummy goal
            actions[:, :2] = normalize_data(actions[:, :2], self.ACTION_STATS)
            delta = get_delta_np(actions)

            return (
                torch.tensor([i], dtype=torch.float32), # for logging purposes
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(pred_image, dtype=torch.float32),
                torch.as_tensor(delta, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)
        
class TrajectoryEvalDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int, 
        context_size: int,
        transform: object,
        traj_names: str,
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        super().__init__(data_folder, data_split_folder, dataset_name, image_size, min_dist_cat, max_dist_cat,
            len_traj_pred, traj_stride, context_size, transform, traj_names, normalize, predefined_index, goals_per_obs)

   
    def _sample_goal(self, trajectory_name, curr_time, min_goal_dist, max_goal_dist):
        """
        Sample a goal from the future in the same trajectory.
        Returns: (trajectory_name, goal_time, goal_is_negative)
        """
        goal_offset = np.random.randint(min_goal_dist, max_goal_dist + 1)
        goal_time = curr_time + int(goal_offset)
        return trajectory_name, goal_time, False

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            f_curr, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[i]
            f_goal, goal_time, _ = self._sample_goal(f_curr, curr_time, min_goal_dist, max_goal_dist)

            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))           
            context = [(f_curr, t) for t in context_times]

            obs_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])
            goal_image = self.transform(Image.open(get_data_path(self.data_folder, f_goal, goal_time))).unsqueeze(0)
            curr_traj_data = self._get_trajectory(f_curr)

            actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, np.array([goal_time]))

            return (
                torch.tensor([i], dtype=torch.float32), # for logging purposes
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(goal_image, dtype=torch.float32),
                torch.as_tensor(actions, dtype=torch.float32),
                torch.as_tensor(goal_pos, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)

class SecondFrameTrainingDataset(BaseDataset):
    """
    Single-step dataset:
      For each trajectory, always use:
        context frame: t=0
        target frame : t=1

    Returns:
      x:         [2, 3, H, W]          (frame0, frame1)
      y:         [3]                   (goal_pos for t=1, in local coords wrt t=0)
      rel_t:     [] or [1]             (scalar, default 1/128)
      sat_img:   [3, Hs, Ws]
      cam2world: [4, 4]                (T_wc at t=1)
      intrinsics:[3, 3]                (K at t=1)

    Notes:
      - This matches your "predict second frame given its camera pose" training.
      - Make sure config has context_size=1 and len_traj_pred=1 when using this dataset.
    """

    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        transform: object,
        traj_names: str = "traj_names.txt",
        normalize: bool = True,
        predefined_index: list = None,
        traj_stride: int = 1,
        # keep these for BaseDataset signature compatibility, but they are ignored:
        min_dist_cat: int = 1,
        max_dist_cat: int = 1,
        len_traj_pred: int = 1,
        context_size: int = 1,
        goals_per_obs: int = 1,
    ):
        super().__init__(
            data_folder=data_folder,
            data_split_folder=data_split_folder,
            dataset_name=dataset_name,
            image_size=image_size,
            min_dist_cat=min_dist_cat,
            max_dist_cat=max_dist_cat,
            len_traj_pred=len_traj_pred,
            traj_stride=traj_stride,
            context_size=context_size,
            transform=transform,
            traj_names=traj_names,
            normalize=normalize,
            predefined_index=predefined_index,
            goals_per_obs=goals_per_obs,
        )

        # Force the intended setup (helps catch config mismatch early)
        self.context_size = 1
        self.len_traj_pred = 1
        self.goals_per_obs = 1

    def _build_index(self, use_tqdm: bool = False):
        """
        Build index: one sample per trajectory (only if it has >=2 frames AND metadata is valid).
        Store tuples: (traj_name, curr_time=0)
        """
        samples_index = []
        goals_index = []

        skipped_traj = 0
        skipped_meta = 0
        kept = 0

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            # 1) 读 traj_data，保证至少两帧
            try:
                traj_data = self._get_trajectory(traj_name)
                traj_len = len(traj_data["position"])
            except Exception:
                skipped_traj += 1
                continue

            if traj_len < 2:
                skipped_traj += 1
                continue

            # 2) 读 meta 并验证：records 非空、且至少有 t=1 的 camera 参数
            try:
                meta = self._get_meta(traj_name)
            except Exception:
                skipped_meta += 1
                continue

            records = meta.get("records", [])
            if not isinstance(records, list) or len(records) < 2:
                # len < 2 意味着 t=1 不存在
                skipped_meta += 1
                continue

            # 检查 t=1 这一帧的 camera 参数是否存在
            try:
                rec1 = records[1]
                cam = rec1.get("camera", None)
                if not isinstance(cam, dict):
                    skipped_meta += 1
                    continue
                K = cam.get("K", None)
                T_wc = cam.get("T_wc", None)
                if K is None or T_wc is None:
                    skipped_meta += 1
                    continue
                # 形状粗检
                if not (len(K) == 3 and len(K[0]) == 3 and len(K[1]) == 3 and len(K[2]) == 3):
                    skipped_meta += 1
                    continue
                if not (len(T_wc) == 4 and len(T_wc[0]) == 4 and len(T_wc[1]) == 4 and len(T_wc[2]) == 4 and len(T_wc[3]) == 4):
                    skipped_meta += 1
                    continue
            except Exception:
                skipped_meta += 1
                continue

            # 3) 合格：一条轨迹一个样本
            samples_index.append((traj_name, 0))
            goals_index.append((traj_name, 1))
            kept += 1

        print(f"[Index:SecondFrame] kept trajs: {kept}, skipped_meta: {skipped_meta}, skipped_traj: {skipped_traj}")
        return samples_index, goals_index

    # def _build_index(self, use_tqdm: bool = False):
    #     """
    #     Build index: one sample per trajectory (only if it has >=2 frames).
    #     Store tuples: (traj_name, curr_time=0)
    #     """
    #     samples_index = []
    #     goals_index = []

    #     for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
    #         # We need at least 2 frames: t=0 and t=1
    #         try:
    #             traj_data = self._get_trajectory(traj_name)
    #             traj_len = len(traj_data["position"])
    #         except Exception:
    #             continue

    #         if traj_len < 2:
    #             continue

    #         # One sample per trajectory at curr_time=0
    #         samples_index.append((traj_name, 0))
    #         goals_index.append((traj_name, 1))

    #     return samples_index, goals_index

    def __getitem__(self, i: int):
        """
        Returns:
          x: [2,3,H,W] (t=0, t=1)
          y: [3]
          rel_t: scalar
          sat_img: [3,Hs,Ws]
          cam2world: [4,4] for t=1
          intrinsics: [3,3] for t=1
        """
        try:
            traj_name, curr_time = self.index_to_data[i]
            t0 = 0
            t1 = 1

            # --- load two frames ---
            img0 = self.transform(Image.open(get_data_path(self.data_folder, traj_name, t0)))
            img1 = self.transform(Image.open(get_data_path(self.data_folder, traj_name, t1)))
            x = torch.stack([img0, img1], dim=0)  # [2,3,H,W]

            # --- action/goal_pos: local goal at t1 wrt t0 ---
            traj_data = self._get_trajectory(traj_name)

            # goal_time expects np array in your original implementation
            _, goal_pos = self._compute_actions(traj_data, curr_time=t0, goal_time=np.array([t1], dtype=np.int32))
            # goal_pos: [1,3] -> [3]
            goal_pos = torch.as_tensor(goal_pos[0], dtype=torch.float32)

            # normalize xy as before
            goal_pos_xy = goal_pos[0:2].unsqueeze(0).numpy()
            goal_pos_xy = normalize_data(goal_pos_xy, self.ACTION_STATS)
            goal_pos[0:2] = torch.from_numpy(goal_pos_xy[0]).to(goal_pos.dtype)

            # --- rel_t: fixed single-step ---
            rel_t = torch.tensor(1.0 / 128.0, dtype=torch.float32)

            # --- satellite image (per trajectory) ---
            sat_img = self._get_satellite_image(traj_name)  # [3,Hs,Ws]
            sat_img = torch.as_tensor(sat_img, dtype=torch.float32)

            # --- camera params at t=1 ---
            meta = self._get_meta(traj_name)
            K, T_wc = self._get_camera_params(meta, t1)
            intrinsics = K.contiguous()         # [3,3]
            cam2world = T_wc.contiguous()       # [4,4]

            return (
                torch.as_tensor(x, dtype=torch.float32),
                goal_pos,          # [3]
                rel_t,             # scalar
                sat_img,           # [3,Hs,Ws]
                cam2world,         # [4,4]
                intrinsics,        # [3,3]
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}:", e)
            raise