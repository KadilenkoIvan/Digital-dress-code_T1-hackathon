"""
YOLO11 Сегментация в реальном времени с веб-камеры
Использует последнюю версию YOLO11-seg для сегментации объектов
Автоматически выбирает GPU если доступен
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch


def main():
    # Загружаем модель YOLO11-seg (последняя версия)
    # Доступные модели: yolo11n-seg.pt, yolo11s-seg.pt, yolo11m-seg.pt, yolo11l-seg.pt, yolo11x-seg.pt
    # n - nano (самая быстрая), s - small, m - medium, l - large, x - xlarge (самая точная)
    model_path = "yolo11n-seg.pt"
    
    # Определяем устройство (автоматически выбирает GPU если доступен)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Загрузка модели {model_path}...")
    print(f"Устройство: {device.upper()}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model = YOLO(model_path)
    model.to(device)
    
    # Открываем веб-камеру
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть веб-камеру")
        return
    
    # Устанавливаем разрешение
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Запуск обнаружения. Нажмите 'q' для выхода")
    
    # Для подсчета времени обработки
    frame_times = []
    
    while True:
        start_time = time.time()
        
        # Читаем кадр
        ret, frame = cap.read()
        
        if not ret:
            print("Ошибка: Не удалось получить кадр")
            break
        
        # Выполняем сегментацию
        results = model(frame, device=device, verbose=False)
        
        # Отрисовываем результаты на кадре
        annotated_frame = results[0].plot()
        
        # Время обработки кадра
        end_time = time.time()
        frame_time_ms = (end_time - start_time) * 1000
        frame_times.append(frame_time_ms)
        if len(frame_times) > 30:
            frame_times.pop(0)
        avg_frame_time = sum(frame_times) / len(frame_times)
        avg_fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0
        
        # Отображаем метрики
        cv2.putText(annotated_frame, f"Time: {avg_frame_time:.1f}ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", (10, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Отображаем количество обнаруженных объектов
        num_objects = len(results[0].boxes) if results[0].boxes is not None else 0
        cv2.putText(annotated_frame, f"Objects: {num_objects}", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Показываем результат
        cv2.imshow('YOLO Segmentation', annotated_frame)
        
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nСреднее время обработки: {avg_frame_time:.1f}ms")
    print(f"Средний FPS: {avg_fps:.1f}")


if __name__ == "__main__":
    main()

