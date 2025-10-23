"""
YOLO11 Segmentation через ONNX Runtime
Использует Ultralytics YOLO с ONNX backend для максимальной производительности
"""
import cv2
import numpy as np
import time
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Путь к модели')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=None, help='Ширина камеры (меньше = быстрее)')
    parser.add_argument('--height', type=int, default=None, help='Высота камеры (меньше = быстрее)')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    
    print("="*70)
    print("YOLO11 OPTIMIZED INFERENCE")
    print("="*70)
    
    # Проверка ONNX Runtime
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"ONNX Runtime провайдеры: {providers}")
        
        if args.device == 'cuda' and 'CUDAExecutionProvider' not in providers:
            print("⚠️ CUDAExecutionProvider недоступен!")
            print("Установите: pip uninstall onnxruntime && pip install onnxruntime-gpu")
    except:
        pass
    
    # Загрузка ONNX модели через Ultralytics
    print(f"Загрузка: {args.model}")
    print(f"Запрошено устройство: {args.device.upper()}")
    
    # Определяем формат модели по пути
    model_format = "ONNX"
    if ".engine" in args.model:
        model_format = "TensorRT"
    elif "openvino" in args.model or "_openvino" in args.model:
        model_format = "OpenVINO"
    
    # Пытаемся определить точность по имени файла (не 100% надежно)
    model_precision = "?"
    model_name_lower = args.model.lower()
    
    if "_half" in model_name_lower or "fp16" in model_name_lower or "_fp16" in model_name_lower:
        model_precision = "FP16"
    elif "int8" in model_name_lower or "_int8" in model_name_lower:
        model_precision = "INT8"
    elif "_fp32" in model_name_lower or model_format == "ONNX":
        model_precision = "FP32"
    
    print(f"Формат: {model_format}")
    print(f"Точность (по имени файла): {model_precision}")
    if model_precision == "?":
        print("⚠️ Тип квантования неизвестен - называйте файлы явно (fp16/int8)")
    print(f"ℹ️ Квантование задается при экспорте, не при запуске!")
    
    try:
        model = YOLO(args.model, task='segment')
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("\nЭкспортируйте модель:")
        print("  ONNX:     python export_to_onnx.py --model yolo11n-seg.pt --format onnx")
        print("  TensorRT: python export_to_onnx.py --model yolo11n-seg.pt --format engine --half")
        print("  OpenVINO: python export_to_onnx.py --model yolo11n-seg.pt --format openvino")
        return
    
    print(f"✅ Модель загружена")
    
    # Камера
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌ Камера {args.camera} недоступна")
        return
    
    # Установка разрешения камеры
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cam = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    print(f"Камера: {w}x{h} @ {fps_cam:.1f}fps")
    
    # Video writer
    writer = None
    if args.save:
        out_file = f"output_onnx_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), fps_cam, (w, h))
        print(f"Сохранение: {out_file}")
    
    print("\nУправление: 'q' - выход, 's' - скриншот")
    print("="*70 + "\n")
    
    # Прогрев
    print("Прогрев модели...")
    dummy = np.zeros((h, w, 3), dtype=np.uint8)
    warmup_times = []
    for i in range(3):
        t_start = time.time()
        _ = model(dummy, conf=args.conf, iou=args.iou, device=args.device, verbose=False)
        warmup_times.append((time.time() - t_start) * 1000)
    print(f"✅ Готово (прогрев: {warmup_times[-1]:.1f}ms)\n")
    
    # Метрики
    times = []
    count = 0
    shots = 0
    avg_time = 0.0
    avg_fps = 0.0
    
    while True:
        t0 = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("❌ Ошибка чтения кадра")
            break
        
        # Inference с явным указанием device
        results = model(frame, conf=args.conf, iou=args.iou, device=args.device, verbose=False)
        
        # Отрисовка
        annotated = results[0].plot()
        
        # Время
        t1 = time.time()
        frame_time = (t1 - t0) * 1000
        times.append(frame_time)
        if len(times) > 30:
            times.pop(0)
        avg_time = sum(times) / len(times)
        avg_fps = 1000 / avg_time if avg_time > 0 else 0
        
        # Метрики
        num_obj = len(results[0].boxes) if results[0].boxes is not None else 0
        
        cv2.putText(annotated, f"Time: {avg_time:.1f}ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, f"{model_format} {model_precision}", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated, f"Device: {args.device.upper()}", (10, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated, f"Objects: {num_obj}", (10, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow(f'YOLO11 {model_format}', annotated)
        
        if writer:
            writer.write(annotated)
        
        count += 1
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fname, annotated)
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
        print(f"Среднее время: {avg_time:.1f}ms")
        print(f"Средний FPS: {avg_fps:.1f}")
    print(f"Кадров: {count}")
    print(f"Скриншотов: {shots}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
