import cv2
import numpy as np
import time
import onnxruntime as ort

# Инициализация ONNX Runtime с GPU
providers = [
    'CUDAExecutionProvider',
    'CPUExecutionProvider'  # Резервный провайдер
]

try:
    sess = ort.InferenceSession("rvm_mobilenetv3_fp16.onnx", providers=providers)
    print("Using GPU (CUDA)")
except Exception as e:
    print(f"CUDA not available: {e}")
    print("Falling back to CPU")
    sess = ort.InferenceSession("your_model.onnx", providers=['CPUExecutionProvider'])

# Получение информации о провайдере
print("Available providers:", ort.get_available_providers())
print("Using provider:", sess.get_providers())

# Получение информации о входных данных модели
input_info = sess.get_inputs()
print("Input info:", {input.name: input.shape for input in input_info})

# Проверяем тип данных для входных тензоров
for input in input_info:
    print(f"Input {input.name}: type {input.type}")

# Инициализация камеры
cap = cv2.VideoCapture(0)  # 0 для встроенной камеры
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Подготовка начальных состояний (rec) - используем float16 как требует модель
rec = [np.zeros([1, 1, 1, 1], dtype=np.float16)] * 4
downsample_ratio = np.array([0.25], dtype=np.float32)

# Загрузка фона
background = cv2.imread('background.jpg')  # Укажи путь к своему фону
if background is None:
    # Если фон не загружен, создаем черный фон
    print("Background not found, using black background")
    background = np.zeros((480, 640, 3), dtype=np.uint8)
else:
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

print("Starting video processing...")

frame_count = 0
total_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    start_time = time.time()

    # Предобработка кадра
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Ресайз если нужно (опционально, для производительности)
    # frame_rgb = cv2.resize(frame_rgb, (640, 480))
    
    # Приведение к формату [B, C, H, W] и нормализация + преобразование в float16
    frame_tensor = frame_rgb.astype(np.float16) / 255.0  # Используем float16
    frame_tensor = frame_tensor.transpose(2, 0, 1)  # HWC to CHW
    frame_tensor = np.expand_dims(frame_tensor, 0)  # Добавляем batch dimension [1, 3, H, W]
    
    #print(f"Input shape: {frame_tensor.shape}, dtype: {frame_tensor.dtype}")

    try:
        # Запуск модели ONNX
        outputs = sess.run(
            None,  # Все выходы
            {
                'src': frame_tensor, 
                'r1i': rec[0], 
                'r2i': rec[1], 
                'r3i': rec[2], 
                'r4i': rec[3], 
                'downsample_ratio': downsample_ratio
            }
        )

        # Извлечение результатов (предполагаем, что первые два выхода - fgr и pha)
        fgr = outputs[0]  # [1, 3, H, W]
        pha = outputs[1]  # [1, 1, H, W]
        
        # Обновление состояний rec (предполагаем, что остальные выходы - это новые состояния)
        if len(outputs) > 2:
            rec = []
            for i in range(2, len(outputs)):
                rec.append(outputs[i].astype(np.float16))

        # Постобработка
        fgr = fgr[0].transpose(1, 2, 0)  # [H, W, 3]
        pha = pha[0].transpose(1, 2, 0)  # [H, W, 1]
        
        # Конвертируем обратно в float32 для дальнейшей обработки
        fgr = fgr.astype(np.float32)
        pha = pha.astype(np.float32)

        # Если pha имеет 1 канал, расширяем до 3 каналов для совместимости
        if pha.shape[2] == 1:
            pha = np.repeat(pha, 3, axis=2)

        # Нормализация значений
        fgr = np.clip(fgr, 0, 1)
        pha = np.clip(pha, 0, 1)

        # Наложение на фон
        background_resized = cv2.resize(background, (fgr.shape[1], fgr.shape[0]))
        background_resized = background_resized.astype(np.float32) / 255.0
        
        comp = fgr * pha + background_resized * (1 - pha)
        comp_bgr = cv2.cvtColor((comp * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Расчет FPS
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
        
        # Отображение оригинального кадра и результата
        #cv2.imshow('Original', frame)
        cv2.imshow('Virtual Background', comp_bgr)

    except Exception as e:
        print(f"Error during inference: {e}")
        break

    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()