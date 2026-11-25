import time
import cv2
from ultralytics import YOLO

# Load the Small YOLO26 Pose model (auto-downloads ~22 MB on first run)
model = YOLO("yolo26s-pose.pt")

cap = cv2.VideoCapture(0)
window_name = "Step 1: YOLO26s Skeleton Tracking"
prev_time = time.time()

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        # Run YOLO26s pose detection on people only (classes=[0])
        results = model(frame, imgsz=320, classes=[0], verbose=False)

        # Draw clean skeleton lines and joints (no bounding box)
        annotated_frame = results[0].plot(boxes=False)

        # Calculate and display live FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(
            annotated_frame,
            f"YOLO26s | FPS: {int(fps)}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 140),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(window_name, annotated_frame)

        # Press 'q' or 'Esc' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

        # Or click the window's 'X' button to quit cleanly
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed cleanly.")