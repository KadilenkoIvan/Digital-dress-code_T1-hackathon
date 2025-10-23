"""
Расширенная версия YOLO сегментации с дополнительными настройками
Позволяет настраивать уверенность, IOU и отображение масок
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Segmentation с веб-камеры')
    parser.add_argument('--model', type=str, default='yolov8n-seg.pt',
                        help='Путь к модели YOLO (по умолчанию: yolov8n-seg.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Порог уверенности (по умолчанию: 0.25)')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='Порог IOU для NMS (по умолчанию: 0.7)')
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
    
    # Загружаем модель
    print(f"Загрузка модели {args.model}...")
    try:
        model = YOLO(args.model)
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
    print(f"  Уверенность: {args.conf}")
    print(f"  IOU: {args.iou}")
    print(f"\nУправление:")
    print("  'q' - выход")
    print("  's' - сделать скриншот")
    print("  '+' - увеличить порог уверенности")
    print("  '-' - уменьшить порог уверенности")
    
    # Переменные для FPS и динамической уверенности
    fps_counter = []
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
        results = model(frame, conf=current_conf, iou=args.iou, verbose=False)
        
        # Отрисовываем результаты
        annotated_frame = results[0].plot()
        
        # Подсчет FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        fps_counter.append(fps)
        if len(fps_counter) > 30:
            fps_counter.pop(0)
        avg_fps = sum(fps_counter) / len(fps_counter)
        
        # Информационная панель
        info_y = 30
        cv2.putText(annotated_frame, f"FPS: {avg_fps:.2f}", (10, info_y), 
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
    print(f"  Средний FPS: {avg_fps:.2f}")
    print(f"  Сделано скриншотов: {screenshot_counter}")


if __name__ == "__main__":
    main()

