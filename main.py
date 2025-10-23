import torch
import cv2
import numpy as np
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('PeterL1n/RobustVideoMatting', 'mobilenetv3')
model = model.to(device).eval()
print(device)

rec = [None] * 4
cap = cv2.VideoCapture(0)

green_color = (0, 255, 0)  # BGR
background = np.full((480, 640, 3), green_color, dtype=np.uint8)

frame_count = 0
total_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    frame_tensor = frame_tensor.to(device)

    with torch.no_grad():
        fgr, pha, *rec = model(frame_tensor, *rec, downsample_ratio=0.25)

    fgr = fgr[0].permute(1, 2, 0).cpu().numpy()
    pha = pha[0].permute(1, 2, 0).cpu().numpy()

    background_resized = cv2.resize(background, (fgr.shape[1], fgr.shape[0]))
    comp = fgr * pha + background_resized / 255 * (1 - pha)
    comp_bgr = cv2.cvtColor((comp * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    frame_time = time.time() - start_time
    total_time += frame_time
    frame_count += 1
    fps = 1.0 / frame_time if frame_time > 0 else 0.0
    avg_fps = frame_count / total_time if total_time > 0 else 0.0
    frame_time *= 1000

    cv2.putText(comp_bgr, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(comp_bgr, f"Avg FPS: {avg_fps:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(comp_bgr, f"Frame time: {frame_time:.2f} ms", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("RVM Green Background", comp_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()