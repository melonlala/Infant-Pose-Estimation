"""Shared helpers for the SMIL synthetic-data tools (keypoint maps, model, projection)."""
import os.path as osp
import pickle
import sys

import numpy as np

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))          # syn_generation/
PROJECT = osp.dirname(ROOT)                                       # hands_on_project/
sys.path.insert(0, osp.join(ROOT, 'smplifyx'))

COCO17_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
COCO17_SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
                   [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
                   [4, 6], [5, 7]]  # 1-based, same as the SyRIP json
# OpenPose COCO-19 order expected by smplifyx (data_parser, utils.smpl_to_openpose 'coco19'):
OP19_NAMES = ['nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder',
              'left_elbow', 'left_wrist', 'mid_hip', 'right_hip', 'right_knee', 'right_ankle',
              'left_hip', 'left_knee', 'left_ankle', 'right_eye', 'left_eye', 'right_ear', 'left_ear']
# SMIL joint index (24 SMPL-topology joints + extra vertices: 24 nose, 25 reye, 26 leye, 27 rear,
# 28 lear; see smplifyx/vertex_joint_selector.py, vertex_ids['smplh']) for each COCO-17 keypoint.
SMIL_FOR_COCO17 = np.array([24, 26, 25, 28, 27, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8])
# joints that are mesh-surface points (face vertices) vs. internal skeleton joints
SURFACE_COCO17 = np.array([1, 1, 1, 1, 1] + [0] * 12, dtype=bool)
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
FOCAL = 1800.0  # cfg_files/fit_smil.yaml


def coco17_to_op19(kp17):
    """(17,3) COCO [x,y,v] -> (19,3) OpenPose COCO-19 [x,y,conf]; conf = COCO visibility (0/1/2),
    the convention of the shipped examples (data/keypoints/*.json use 2.0)."""
    kp = np.asarray(kp17, dtype=np.float64).reshape(17, 3)
    idx = {n: i for i, n in enumerate(COCO17_NAMES)}
    out = np.zeros((19, 3))
    for j, name in enumerate(OP19_NAMES):
        if name in idx:
            out[j] = kp[idx[name]]
    for j, (a, b) in {1: ('left_shoulder', 'right_shoulder'), 8: ('left_hip', 'right_hip')}.items():
        pa, pb = kp[idx[a]], kp[idx[b]]
        if pa[2] > 0 and pb[2] > 0:
            out[j] = [(pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, min(pa[2], pb[2])]
    out[out[:, 2] <= 0] = 0.0
    return out


class SMILNumpy:
    """Minimal numpy SMIL forward pass (rodrigues + LBS) on models/smil_web_np.npz."""

    def __init__(self, path=osp.join(ROOT, 'models', 'smil_web_np.npz')):
        d = np.load(path)
        self.v_template, self.shapedirs, self.posedirs = d['v_template'], d['shapedirs'], d['posedirs']
        self.J_regressor, self.weights, self.f = d['J_regressor'], d['weights'], d['f']
        self.parents = d['kintree_table'][0].copy()
        self.parents[0] = -1
        from vertex_ids import vertex_ids
        vi = vertex_ids['smplh']
        self.extra = np.array([vi[k] for k in ['nose', 'reye', 'leye', 'rear', 'lear']])

    @staticmethod
    def rodrigues(r):
        r = r.reshape(-1, 3)
        th = np.linalg.norm(r + 1e-8, axis=1, keepdims=True)
        k = r / th
        K = np.zeros((len(r), 3, 3))
        K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
        K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
        s, c = np.sin(th)[:, :, None], np.cos(th)[:, :, None]
        return np.eye(3)[None] + s * K + (1 - c) * K @ K

    def __call__(self, betas, global_orient, body_pose):
        pose = np.concatenate([np.ravel(global_orient), np.ravel(body_pose)])
        betas = np.ravel(betas)
        v_shaped = self.v_template + self.shapedirs @ betas
        J = self.J_regressor @ v_shaped
        R = self.rodrigues(pose)
        pose_feat = (R[1:] - np.eye(3)).ravel()
        v_posed = v_shaped + self.posedirs @ pose_feat
        G = np.zeros((24, 4, 4))
        for i in range(24):
            T = np.eye(4)
            T[:3, :3] = R[i]
            T[:3, 3] = J[i] - (J[self.parents[i]] if i > 0 else 0)
            G[i] = T if i == 0 else G[self.parents[i]] @ T
        Jt = G[:, :3, 3].copy()
        Gr = G.copy()
        Gr[:, :3, 3] -= np.einsum('kij,kj->ki', G[:, :3, :3], J)
        Tv = np.einsum('vk,kij->vij', self.weights, Gr)
        verts = np.einsum('vij,vj->vi', Tv[:, :3, :3], v_posed) + Tv[:, :3, 3]
        joints = np.concatenate([Jt, verts[self.extra]], 0)   # (29,3)
        return verts, joints


def load_fit(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def project(points, R=np.eye(3), t=np.zeros(3), focal=FOCAL, center=(0, 0)):
    """Same pinhole model as smplifyx/camera.py: x_img = f * (Rp+t)_xy / (Rp+t)_z + c."""
    pc = points @ np.asarray(R).T + np.ravel(t)
    return focal * pc[:, :2] / pc[:, 2:3] + np.asarray(center)[None], pc[:, 2]


def load_hip_offset(path=osp.join(ROOT, 'models', 'coco_hip_offset.json')):
    """3D offset (root frame, m) of the COCO-style left hip from the SMIL hip joint; None if absent.
    Written by tools/fit_hip_offset.py."""
    if not osp.exists(path):
        return None
    import json
    return np.array(json.load(open(path))['offset_left_m'])


def coco17_points(joints, global_orient, hip_offset=None):
    """(17,3) 3D COCO keypoints from SMIL joints (same frame as `joints`, i.e. before any
    camera/view transform). With hip_offset, hips = J_hip + R(global_orient) @ (+-ox, oy, oz)."""
    P = joints[SMIL_FOR_COCO17].copy()
    if hip_offset is not None:
        R = SMILNumpy.rodrigues(np.ravel(global_orient))[0]
        P[11] = joints[1] + R @ hip_offset
        P[12] = joints[2] + R @ (hip_offset * [-1, 1, 1])
    return P
