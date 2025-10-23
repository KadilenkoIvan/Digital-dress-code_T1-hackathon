"""
Оптимизированная версия YOLO11 с использованием встроенных оптимизаций Ultralytics
Использует INT8/FP16 квантизацию и другие методы ускорения
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch
import argparse


def main():
    parser = argparse.ArgumentParser(description='YOLO11 Optimized Inference')
    parser.add_argument('--model', type=str, default='yolo11n-seg.pt',
                        help='Путь к модели (по умолчанию: yolo11n-seg.pt)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Устройство (по умолчанию: auto)')
    parser.add_argument('--half', action='store_true',
                        help='Использовать FP16 (только GPU, ускорение ~2x)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Порог уверенности (по умолчанию: 0.25)')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='Порог IOU (по умолчанию: 0.7)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Размер входного изображения - меньше = быстрее (по умолчанию: 640)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Индекс камеры (по умолчанию: 0)')
    parser.add_argument('--save', action='store_true',
                        help='Сохранять видео')
    
    args = parser.parse_args()
    
    # Определяем устройство
    if args.device == 'auto':
        device = 'cpu' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # FP16 только на GPU
    use_half = args.half and device == 'cuda'
    
    print("="*70)
    print("YOLO11 OPTIMIZED INFERENCE - Максимальная производительность")
    print("="*70)
    print(f"Модель: {args.model}")
    print(f"Устройство: {device.upper()}")
    print(f"FP16: {use_half}")
    print(f"Размер изображения: {args.imgsz} (меньше = быстрее)")
    print(f"Уверенность: {args.conf}")
    print(f"IOU: {args.iou}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("="*70)
    
    # Загрузка модели
    print("\nЗагрузка модели...")
    model = YOLO(args.model)
    
    # FP16 будет применен автоматически через параметр half при inference
    if use_half:
        print("FP16 будет применен автоматически при inference")
    
    # Открываем камеру
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"❌ Не удалось открыть камеру {args.camera}")
        return
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    print(f"Разрешение камеры: {actual_width}x{actual_height}")
    print(f"FPS камеры: {camera_fps}")
    
    # Видео writer
    video_writer = None
    if args.save:
        import time as tm
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_filename = f"output_optimized_{tm.strftime('%Y%m%d_%H%M%S')}.mp4"
        video_writer = cv2.VideoWriter(output_filename, fourcc, camera_fps, 
                                       (actual_width, actual_height))
        print(f"Сохранение видео: {output_filename}")
    
    print("\nНажмите 'q' для выхода, 's' для скриншота")
    print("="*70 + "\n")
    
    frame_times = []
    frame_count = 0
    screenshot_counter = 0
    avg_frame_time = 0.0
    avg_fps = 0.0
    
    # Прогрев модели (важно для корректного замера времени)
    print("Прогрев модели...")
    dummy_frame = np.zeros((actual_height, actual_width, 3), dtype=np.uint8)
    for _ in range(5):
        _ = model(dummy_frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, 
                  device=device, verbose=False)
    print("✅ Готово к работе\n")
    
    while True:
        
        
        ret, frame = cap.read()
        if not ret:
            print("❌ Ошибка чтения кадра")
            break
        
        start_time = time.time()
        # Inference с оптимизациями
        results = model(
            frame, 
            imgsz=args.imgsz,  # Размер для inference
            conf=args.conf,
            iou=args.iou,
            device=device,
            half=use_half,
            verbose=False,
            agnostic_nms=True,  # Класс-агностик NMS (быстрее)
            max_det=100  # Максимум детекций (меньше = быстрее)
        )

        result = results[0]

        mask_total = np.zeros(frame.shape[:2], dtype=np.uint8)

        if result.masks is not None:
            for box, mask, cls in zip(result.boxes.xyxy, result.masks.data, result.boxes.cls):
                if int(cls) == 0:
                    mask = mask.cpu().numpy()
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    mask = (mask > 0.5).astype(np.uint8)
                    mask_total = np.maximum(mask_total, mask)
        
            annotated_frame = cv2.bitwise_and(frame, frame, mask=mask_total)
        else:
            annotated_frame = results[0].plot()

        end_time = time.time()
        frame_time_ms = (end_time - start_time) * 1000
        frame_times.append(frame_time_ms)
        # Отрисовка
        #annotated_frame = results[0].plot()
        
        # Время обработки
        
        if len(frame_times) > 30:
            frame_times.pop(0)
        avg_frame_time = sum(frame_times) / len(frame_times)
        avg_fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0
        
        # Информация
        num_objects = len(results[0].boxes) if results[0].boxes is not None else 0
        
        cv2.putText(annotated_frame, f"Time: {avg_frame_time:.1f}ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f} (Optimized)", (10, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Device: {device.upper()}" + (" FP16" if use_half else ""), 
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Objects: {num_objects}", (10, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"ImgSz: {args.imgsz}", (10, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('YOLO11 Optimized', annotated_frame)
        
        if video_writer is not None:
            video_writer.write(annotated_frame)
        
        frame_count += 1
        
        # Клавиши
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            import time as tm
            screenshot_filename = f"screenshot_opt_{tm.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(screenshot_filename, annotated_frame)
            print(f"📸 Скриншот сохранен: {screenshot_filename}")
            screenshot_counter += 1
    
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    
    print(f"\n{'='*70}")
    print("СТАТИСТИКА")
    print(f"{'='*70}")
    if frame_count > 0:
        print(f"Среднее время обработки: {avg_frame_time:.1f}ms")
        print(f"Средний FPS: {avg_fps:.1f}")
    else:
        print("Кадры не обработаны")
    print(f"Обработано кадров: {frame_count}")
    print(f"Скриншотов: {screenshot_counter}")
    if video_writer is not None:
        print(f"Видео сохранено")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

