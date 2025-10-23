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
    
    while True:
        # Читаем кадр с камеры
        ret, frame = cap.read()
        
        if not ret:
            print("Ошибка: Не удалось получить кадр")
            break
        
        # Отображаем FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
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

