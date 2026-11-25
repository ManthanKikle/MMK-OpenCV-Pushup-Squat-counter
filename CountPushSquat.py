import time
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO26s Pose model
model = YOLO("yolo26s-pose.pt")

# COCO 17-Keypoint Indices
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16


def calculate_angle(a, b, c):
    """Calculate inner angle (0-180 degrees) at joint b."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle


def get_best_side(kpts, confs, left_ids, right_ids, min_conf=0.3):
    """Pick whichever side (left or right) has higher total confidence."""
    left_conf = sum(confs[i] for i in left_ids)
    right_conf = sum(confs[i] for i in right_ids)
    chosen = left_ids if left_conf >= right_conf else right_ids
    if all(confs[i] >= min_conf for i in chosen):
        return [kpts[i] for i in chosen]
    return None


cap = cv2.VideoCapture(0)
window_name = "AI Squat, Pushup & Jump Counter"
prev_time = time.time()

# Smoothing buffer
smooth_kpts = None
ALPHA = 0.6

# Counters and states
squat_count = 0
squat_stage = "UP"

pushup_count = 0
pushup_stage = "UP"

jump_count = 0
jump_stage = "GROUND"
baseline_hip_y = None  # Tracks normal standing hip height

active_mode = "READY"
rep_progress = 0.0

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        # Run YOLO26s pose detection
        results = model(frame, imgsz=480, classes=[0], verbose=False)
        annotated_frame = results[0].plot(boxes=False)

        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            raw_kpts = results[0].keypoints.xy[0].cpu().numpy()
            confs = results[0].keypoints.conf[0].cpu().numpy()

            # Smooth keypoints
            if smooth_kpts is None:
                smooth_kpts = raw_kpts.copy()
            else:
                smooth_kpts = ALPHA * raw_kpts + (1 - ALPHA) * smooth_kpts

            kpts = smooth_kpts

            # Torso dimensions for posture detection
            mid_shoulder = (kpts[LEFT_SHOULDER] + kpts[RIGHT_SHOULDER]) / 2
            mid_hip = (kpts[LEFT_HIP] + kpts[RIGHT_HIP]) / 2
            torso_dx = abs(mid_shoulder[0] - mid_hip[0])
            torso_dy = abs(mid_hip[1] - mid_shoulder[1])
            torso_len = np.linalg.norm(mid_shoulder - mid_hip) + 1e-6

            mid_wrist = (kpts[LEFT_WRIST] + kpts[RIGHT_WRIST]) / 2
            wrist_below_shoulders = (mid_wrist[1] - mid_shoulder[1]) > torso_len * 0.45

            # --- UPRIGHT POSTURE: SQUATS & JUMPS ---
            if torso_dy > torso_dx * 1.1 and not (wrist_below_shoulders and torso_dy < torso_len * 0.75):
                leg_pts = get_best_side(
                    kpts, confs,
                    [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE],
                    [RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE]
                )
                if leg_pts is not None:
                    hip, knee, ankle = leg_pts

                    # 1. Squat Depth Calculation
                    knee_angle = calculate_angle(hip, knee, ankle)
                    thigh_len = np.linalg.norm(hip - knee) + 1e-6
                    hip_knee_vert_ratio = (knee[1] - hip[1]) / thigh_len

                    angle_depth = np.clip((160 - knee_angle) / (160 - 105), 0.0, 1.0)
                    front_depth = np.clip((0.85 - hip_knee_vert_ratio) / (0.85 - 0.35), 0.0, 1.0)
                    rep_progress = max(angle_depth, front_depth)

                    if rep_progress >= 0.80:
                        squat_stage = "DOWN"
                    elif rep_progress <= 0.25 and squat_stage == "DOWN":
                        squat_stage = "UP"
                        squat_count += 1

                    # 2. Jump Detection (tracks upward hip displacement above standing baseline)
                    current_hip_y = mid_hip[1]
                    if baseline_hip_y is None:
                        baseline_hip_y = current_hip_y

                    # Update standing baseline slowly ONLY when standing upright & grounded
                    if squat_stage == "UP" and rep_progress < 0.20 and jump_stage == "GROUND":
                        # If user moves closer/farther or settles, adapt baseline smoothly
                        if abs(current_hip_y - baseline_hip_y) < torso_len * 0.18:
                            baseline_hip_y = 0.90 * baseline_hip_y + 0.10 * current_hip_y

                    # Remember: in image coordinates, smaller Y = higher up in the air!
                    jump_height = baseline_hip_y - current_hip_y
                    jump_threshold = torso_len * 0.22  # Must rise at least ~22% of torso height

                    if jump_height > jump_threshold and squat_stage == "UP":
                        jump_stage = "AIRBORNE"
                        active_mode = "JUMP"
                    elif jump_height <= jump_threshold * 0.45 and jump_stage == "AIRBORNE":
                        jump_stage = "GROUND"
                        jump_count += 1
                        active_mode = "JUMP"
                    else:
                        active_mode = "SQUAT / JUMP"

                    cv2.putText(annotated_frame, f"Depth: {int(rep_progress * 100)}%",
                                (int(knee[0]) + 12, int(knee[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            # --- HORIZONTAL / PLANK POSTURE: PUSHUPS ---
            else:
                baseline_hip_y = None  # Reset standing baseline while on the floor
                jump_stage = "GROUND"

                arm_pts = get_best_side(
                    kpts, confs,
                    [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST],
                    [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST]
                )
                if arm_pts is not None:
                    active_mode = "PUSHUP"
                    shoulder, elbow, wrist = arm_pts

                    elbow_angle = calculate_angle(shoulder, elbow, wrist)
                    arm_len = (np.linalg.norm(shoulder - elbow) + np.linalg.norm(elbow - wrist)) + 1e-6
                    shoulder_wrist_vert_ratio = (wrist[1] - shoulder[1]) / arm_len

                    angle_depth = np.clip((155 - elbow_angle) / (155 - 95), 0.0, 1.0)
                    front_depth = np.clip((0.85 - shoulder_wrist_vert_ratio) / (0.85 - 0.40), 0.0, 1.0)
                    rep_progress = max(angle_depth, front_depth)

                    cv2.putText(annotated_frame, f"Depth: {int(rep_progress * 100)}%",
                                (int(elbow[0]) + 12, int(elbow[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

                    if rep_progress >= 0.80:
                        pushup_stage = "DOWN"
                    elif rep_progress <= 0.25 and pushup_stage == "DOWN":
                        pushup_stage = "UP"
                        pushup_count += 1

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # --- DRAW HUD DASHBOARD ---
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (10, 10), (340, 190), (20, 20, 20), -1)
        annotated_frame = cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0)

        cv2.putText(annotated_frame, f"MODE: {active_mode} | FPS: {int(fps)}", (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"SQUATS:  {squat_count}  [{squat_stage}]", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 140), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"PUSHUPS: {pushup_count}  [{pushup_stage}]", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"JUMPS:   {jump_count}  [{jump_stage}]", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 180, 50), 2, cv2.LINE_AA)

        # --- DRAW LIVE DEPTH PROGRESS BAR ---
        bar_x, bar_y, bar_w, bar_h = 20, 210, 22, 180
        cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_h = int(bar_h * rep_progress)
        bar_color = (0, 255, 140) if rep_progress >= 0.80 else (0, 200, 255)
        cv2.rectangle(annotated_frame, (bar_x, bar_y + bar_h - fill_h), (bar_x + bar_w, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)

        cv2.imshow(window_name, annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("r"):
            squat_count, pushup_count, jump_count = 0, 0, 0
            squat_stage, pushup_stage, jump_stage = "UP", "UP", "GROUND"
            baseline_hip_y = None
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed cleanly.")