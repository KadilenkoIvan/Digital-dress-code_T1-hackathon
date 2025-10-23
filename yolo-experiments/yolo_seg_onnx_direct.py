"""
YOLO11 Segmentation - прямой ONNX Runtime без Ultralytics
Полная поддержка CPU и GPU
"""
import cv2
import numpy as np
import time
import argparse
import onnxruntime as ort


def xywh2xyxy(x):
    """(center_x, center_y, w, h) -> (x1, y1, x2, y2)"""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def nms(boxes, scores, iou_threshold=0.45):
    """Non-Maximum Suppression"""
    if len(boxes) == 0:
        return []
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep


class YOLOv11Seg:
    """YOLO11 Segmentation с прямым ONNX Runtime"""
    
    def __init__(self, model_path, device='cpu'):
        print(f"Инициализация ONNX Runtime...")
        print(f"Модель: {model_path}")
        
        # Провайдеры
        providers = []
        if device == 'cuda':
            providers.append(('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }))
        providers.append('CPUExecutionProvider')
        
        # Сессия
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        
        # Проверка провайдеров
        active = self.session.get_providers()
        print(f"Доступные провайдеры: {ort.get_available_providers()}")
        print(f"Активные провайдеры: {active}")
        
        if device == 'cuda' and 'CUDAExecutionProvider' in active:
            self.device = 'CUDA'
            print("✅ Используется GPU")
        else:
            self.device = 'CPU'
            if device == 'cuda':
                print("⚠️ CUDA недоступна, используется CPU")
        
        # Входы/выходы
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.img_size = self.input_shape[2]
        
        print(f"Входной размер: {self.img_size}x{self.img_size}")
        print(f"Устройство: {self.device}")
        
        # COCO классы
        self.classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Цвета
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, (len(self.classes), 3), dtype=np.uint8)
    
    def preprocess(self, img):
        """Предобработка"""
        self.orig_h, self.orig_w = img.shape[:2]
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_chw = img_norm.transpose(2, 0, 1)
        img_batch = np.expand_dims(img_chw, axis=0).astype(np.float32)
        return img_batch
    
    def postprocess(self, outputs, conf_thresh=0.25, iou_thresh=0.45):
        """Постобработка с масками"""
        # outputs[0]: [1, 116, 8400] - detections
        # outputs[1]: [1, 32, 160, 160] - protos
        
        preds = outputs[0][0].T  # [8400, 116]
        protos = outputs[1][0] if len(outputs) > 1 else None  # [32, 160, 160]
        
        boxes_xywh = preds[:, :4]
        scores = preds[:, 4:84]
        mask_coeffs = preds[:, 84:] if protos is not None else None
        
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        mask = confidences > conf_thresh
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        if mask_coeffs is not None:
            mask_coeffs = mask_coeffs[mask]
        
        if len(boxes_xywh) == 0:
            return [], [], [], []
        
        boxes = xywh2xyxy(boxes_xywh)
        boxes[:, [0, 2]] *= self.orig_w / self.img_size
        boxes[:, [1, 3]] *= self.orig_h / self.img_size
        
        indices = nms(boxes, confidences, iou_thresh)
        
        boxes = boxes[indices]
        confidences = confidences[indices]
        class_ids = class_ids[indices]
        if mask_coeffs is not None:
            mask_coeffs = mask_coeffs[indices]
        
        # Декодирование масок
        masks = []
        if protos is not None and len(indices) > 0:
            for i, coeffs in enumerate(mask_coeffs):
                mask = np.sum(coeffs[:, None, None] * protos, axis=0)
                mask = 1 / (1 + np.exp(-mask))  # sigmoid
                mask = cv2.resize(mask, (self.orig_w, self.orig_h))
                
                x1, y1, x2, y2 = boxes[i].astype(int)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(self.orig_w, x2)
                y2 = min(self.orig_h, y2)
                
                mask_bin = (mask > 0.5).astype(np.uint8)
                mask_cropped = np.zeros((self.orig_h, self.orig_w), dtype=np.uint8)
                mask_cropped[y1:y2, x1:x2] = mask_bin[y1:y2, x1:x2]
                
                masks.append(mask_cropped)
        
        return boxes, confidences, class_ids, masks
    
    def draw(self, img, boxes, confidences, class_ids, masks):
        """Отрисовка"""
        result = img.copy()
        overlay = result.copy()
        
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = box.astype(int)
            color = tuple(map(int, self.colors[cls_id]))
            
            # Маска
            if i < len(masks):
                mask = masks[i]
                overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array(color) * 0.5
            
            # Bbox
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Метка
            label = f"{self.classes[cls_id]}: {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - h - 8), (x1 + w, y1), color, -1)
            cv2.putText(result, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if len(masks) > 0:
            result = cv2.addWeighted(result, 0.6, overlay, 0.4, 0)
        
        return result
    
    def __call__(self, img, conf=0.25, iou=0.45):
        """Inference"""
        input_tensor = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        boxes, confidences, class_ids, masks = self.postprocess(outputs, conf, iou)
        return boxes, confidences, class_ids, masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Путь к .onnx модели')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    
    print("="*70)
    print("YOLO11 ONNX DIRECT - Pure ONNX Runtime")
    print("="*70)
    
    # Модель
    model = YOLOv11Seg(args.model, args.device)
    
    # Камера
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌ Камера {args.camera} недоступна")
        return
    
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cam = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    print(f"Камера: {w}x{h} @ {fps_cam:.1f}fps")
    
    # Writer
    writer = None
    if args.save:
        out = f"output_direct_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps_cam, (w, h))
        print(f"Сохранение: {out}")
    
    print("\nУправление: 'q' - выход, 's' - скриншот")
    print("="*70 + "\n")
    
    # Прогрев
    print("Прогрев...")
    dummy = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(3):
        _ = model(dummy, args.conf, args.iou)
    print("✅ Готово\n")
    
    times = []
    count = 0
    shots = 0
    avg_time = 0.0
    
    while True:
        t0 = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Inference
        boxes, confs, cls_ids, masks = model(frame, args.conf, args.iou)
        
        # Отрисовка
        result = model.draw(frame, boxes, confs, cls_ids, masks)
        
        # Время
        frame_time = (time.time() - t0) * 1000
        times.append(frame_time)
        if len(times) > 30:
            times.pop(0)
        avg_time = sum(times) / len(times)
        avg_fps = 1000 / avg_time if avg_time > 0 else 0
        
        # Метрики
        cv2.putText(result, f"Time: {avg_time:.1f}ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(result, f"FPS: {avg_fps:.1f}", (10, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(result, f"Device: {model.device}", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result, f"Objects: {len(boxes)}", (10, 135), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('YOLO11 ONNX Direct', result)
        
        if writer:
            writer.write(result)
        
        count += 1
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fname, result)
            print(f"📸 {fname}")
            shots += 1
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"\n{'='*70}")
    print("СТАТИСТИКА")
    print(f"{'='*70}")
    if count > 0:
        print(f"Время: {avg_time:.1f}ms")
        print(f"FPS: {avg_fps:.1f}")
    print(f"Кадры: {count}")
    print(f"Скриншоты: {shots}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

