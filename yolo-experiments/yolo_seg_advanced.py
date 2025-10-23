"""
Расширенная версия YOLO11 сегментации с дополнительными настройками
Использует последнюю версию YOLO11 для максимальной производительности
Позволяет настраивать уверенность, IOU, устройство (GPU/CPU) и другие параметры
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO11 Segmentation с веб-камеры')
    parser.add_argument('--model', type=str, default='yolo11n-seg.pt',
                        help='Путь к модели YOLO11 (по умолчанию: yolo11n-seg.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Порог уверенности (по умолчанию: 0.25)')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='Порог IOU для NMS (по умолчанию: 0.7)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'gpu'],
                        help='Устройство для вычислений: auto (авто), cpu, cuda/gpu (по умолчанию: auto)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Индекс камеры (по умолчанию: 0)')
    parser.add_argument('--width', type=int, default=1280,
                        help='Ширина кадра (по умолчанию: 1280)')
    parser.add_argument('--height', type=int, default=720,
                        help='Высота кадра (по умолчанию: 720)')
    parser.add_argument('--save', action='store_true',
                        help='Сохранять видео с результатами')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Определяем устройство
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif args.device in ['gpu', 'cuda']:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            print("⚠️ CUDA недоступна, используется CPU")
            device = 'cpu'
    else:
        device = 'cpu'
    
    # Загружаем модель
    print(f"Загрузка модели {args.model}...")
    print(f"Устройство: {device.upper()}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA версия: {torch.version.cuda}")
    
    try:
        model = YOLO(args.model)
        model.to(device)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        print("Убедитесь, что файл модели существует или будет загружен автоматически")
        return
    
    # Открываем веб-камеру
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть камеру с индексом {args.camera}")
        return
    
    # Устанавливаем разрешение
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Получаем фактическое разрешение
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_fps = cap.get(cv2.CAP_PROP_FPS)
    if camera_fps == 0 or camera_fps > 1200:  # Некоторые камеры возвращают 0 или неверное значение
        camera_fps = 30.0  # Fallback значение
    print(f"Разрешение камеры: {actual_width}x{actual_height}")
    print(f"FPS камеры: {camera_fps}")
    
    # Настройка сохранения видео
    video_writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_filename = f"output_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        video_writer = cv2.VideoWriter(output_filename, fourcc, camera_fps, 
                                       (actual_width, actual_height))
        print(f"Сохранение видео в: {output_filename} ({camera_fps} FPS)")
    
    print(f"\nНастройки:")
    print(f"  Модель: {args.model}")
    print(f"  Устройство: {device.upper()}")
    print(f"  Уверенность: {args.conf}")
    print(f"  IOU: {args.iou}")
    print(f"\nУправление:")
    print("  'q' - выход")
    print("  's' - сделать скриншот")
    print("  '+' - увеличить порог уверенности")
    print("  '-' - уменьшить порог уверенности")
    
    # Переменные для метрик и динамической уверенности
    frame_times = []
    current_conf = args.conf
    screenshot_counter = 0
    
    while True:
        start_time = time.time()
        
        # Читаем кадр
        ret, frame = cap.read()
        
        if not ret:
            print("Ошибка: Не удалось получить кадр")
            break
        
        # Выполняем сегментацию
        results = model(frame, conf=current_conf, iou=args.iou, device=device, verbose=False)
        
        # Отрисовываем результаты
        annotated_frame = results[0].plot()
        
        # Время обработки кадра
        end_time = time.time()
        frame_time_ms = (end_time - start_time) * 1000
        frame_times.append(frame_time_ms)
        if len(frame_times) > 30:
            frame_times.pop(0)
        avg_frame_time = sum(frame_times) / len(frame_times)
        avg_fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0
        
        # Информационная панель
        info_y = 30
        cv2.putText(annotated_frame, f"Time: {avg_frame_time:.1f}ms", (10, info_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        info_y += 35
        cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", (10, info_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        num_objects = len(results[0].boxes) if results[0].boxes is not None else 0
        info_y += 35
        cv2.putText(annotated_frame, f"Objects: {num_objects}", (10, info_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        info_y += 35
        cv2.putText(annotated_frame, f"Conf: {current_conf:.2f}", (10, info_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Показываем результат
        cv2.imshow('YOLO Segmentation Advanced', annotated_frame)
        
        # Сохранение видео
        if video_writer is not None:
            video_writer.write(annotated_frame)
        
        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_filename = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(screenshot_filename, annotated_frame)
            print(f"Скриншот сохранен: {screenshot_filename}")
            screenshot_counter += 1
        elif key == ord('+') or key == ord('='):
            current_conf = min(0.95, current_conf + 0.05)
            print(f"Уверенность: {current_conf:.2f}")
        elif key == ord('-') or key == ord('_'):
            current_conf = max(0.05, current_conf - 0.05)
            print(f"Уверенность: {current_conf:.2f}")
    
    # Освобождаем ресурсы
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    
    print(f"\nСтатистика:")
    print(f"  Среднее время обработки: {avg_frame_time:.1f}ms")
    print(f"  Средний FPS: {avg_fps:.1f}")
    print(f"  Сделано скриншотов: {screenshot_counter}")


if __name__ == "__main__":
    main()

