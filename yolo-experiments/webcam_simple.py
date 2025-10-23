"""
Простой просмотр веб-камеры без обработки
"""
import cv2


def main():
    # Открываем веб-камеру (0 - индекс камеры по умолчанию)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть веб-камеру")
        return
    
    # Устанавливаем разрешение (опционально)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Нажмите 'q' для выхода")
    
    frame_times = []
    
    while True:
        start_time = time.time()
        
        # Читаем кадр с камеры
        ret, frame = cap.read()
        
        if not ret:
            print("Ошибка: Не удалось получить кадр")
            break
        
        # Время обработки кадра
        end_time = time.time()
        frame_time_ms = (end_time - start_time) * 1000
        frame_times.append(frame_time_ms)
        if len(frame_times) > 30:
            frame_times.pop(0)
        avg_frame_time = sum(frame_times) / len(frame_times)
        
        # Отображаем метрики
        cv2.putText(frame, f"Frame time: {avg_frame_time:.1f}ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {1000/avg_frame_time:.1f}", (10, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Показываем кадр
        cv2.imshow('Webcam', frame)
        
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

