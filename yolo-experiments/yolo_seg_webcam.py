"""
YOLO Сегментация в реальном времени с веб-камеры
Использует YOLOv8-seg для сегментации объектов
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch


def main():
    # Загружаем модель YOLO-seg
    # Доступные модели: yolov8n-seg.pt, yolov8s-seg.pt, yolov8m-seg.pt, yolov8l-seg.pt, yolov8x-seg.pt
    # n - nano (самая быстрая), s - small, m - medium, l - large, x - xlarge (самая точная)
    model_path = "yolov8n-seg.pt"
    
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
    
    # Для подсчета FPS
    fps_counter = []
    
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
        
        # Подсчет FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        fps_counter.append(fps)
        if len(fps_counter) > 30:
            fps_counter.pop(0)
        avg_fps = sum(fps_counter) / len(fps_counter)
        
        # Отображаем FPS
        cv2.putText(annotated_frame, f"FPS: {avg_fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Отображаем количество обнаруженных объектов
        num_objects = len(results[0].boxes) if results[0].boxes is not None else 0
        cv2.putText(annotated_frame, f"Objects: {num_objects}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Показываем результат
        cv2.imshow('YOLO Segmentation', annotated_frame)
        
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Средний FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main()

