"""Create source-overlay evidence and numeric diagnostics, not accuracy scores."""
from pathlib import Path
import argparse
import json
import sys
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motionbridge.global_motion import SMPL22_NAMES

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.motion, allow_pickle=False) as data:
        joints = data["joints_camera"]
        world = data["joints_world"]
        K = data["K"]
        parents = data["parents"]
        root = data["world_transl"]
        orient = data["world_global_orient"]
        fps = float(data["fps"])
    pixel_h = np.einsum("fij,fkj->fki", K, joints)
    positive_depth = pixel_h[..., 2] > 1e-5
    pixel = pixel_h[..., :2] / np.maximum(pixel_h[..., 2:], 1e-5)
    cap = cv2.VideoCapture(str(args.video))
    width, height = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(str(args.output / "source_pose_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width*2, height))
    if not writer.isOpened():
        raise RuntimeError("Cannot open video writer")
    samples = []
    sample_frames = set(int(i*fps) for i in range(6))
    try:
        for f in range(len(joints)):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Source video ends at frame {f}")
            overlay = frame.copy()
            for j, p in enumerate(parents):
                if p < 0 or not (positive_depth[f, j] and positive_depth[f, p]):
                    continue
                color = (30,180,255) if SMPL22_NAMES[j].startswith("Left") else (255,210,20) if SMPL22_NAMES[j].startswith("Right") else (255,255,255)
                a = tuple(np.rint(pixel[f,p]).astype(int))
                b = tuple(np.rint(pixel[f,j]).astype(int))
                if max(abs(v) for v in (*a,*b)) < 100000:
                    cv2.line(overlay, a, b, color, 3, cv2.LINE_AA)
                    cv2.circle(overlay, b, 4, color, -1, cv2.LINE_AA)
            cv2.putText(frame, f"Source {f/fps:.2f}s", (10,25), cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
            cv2.putText(overlay, "GVHMR camera reprojection", (10,25), cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
            combined = np.concatenate((frame, overlay), axis=1)
            writer.write(combined)
            if f in sample_frames:
                scale = 480 / combined.shape[1]
                samples.append(cv2.resize(combined, None, fx=scale, fy=scale))
    finally:
        cap.release()
        writer.release()
    if samples:
        cv2.imwrite(str(args.output / "source_pose_contact_sheet.jpg"), np.concatenate(samples, axis=0))
    matrices = Rotation.from_rotvec(orient).as_matrix()
    facing = matrices @ np.array([0.,0.,1.])
    yaw = np.unwrap(np.arctan2(facing[:,0], facing[:,2]))
    report = {"frames":len(joints), "fps":fps,
              "root_translation_range_m":np.ptp(root,axis=0).tolist(),
              "root_yaw_range_deg":float(np.rad2deg(np.ptp(yaw))),
              "samples":[{"time_in_clip_s":f/fps,"world_root_m":root[f].tolist(),
                          "facing_world":facing[f].tolist(),"yaw_deg":float(np.rad2deg(yaw[f]))}
                         for f in sorted(sample_frames) if f < len(root)],
              "max_root_speed_m_s":float(np.linalg.norm(np.diff(root,axis=0),axis=-1).max()*fps),
              "minimum_joint_height_m":float(world[...,1].min()),
              "note":"Diagnostics only: reprojection consistency, facing range and file validity are not motion accuracy. Compare source back-facing frame visually."}
    (args.output / "motion_diagnostics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__ == "__main__":
    main()
