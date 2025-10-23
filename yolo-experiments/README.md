# YOLO Segmentation для веб-камеры

Набор скриптов для работы с YOLO сегментацией в реальном времени.

## Установка

### 1. Установите необходимые библиотеки:

```bash
pip install -r requirements.txt
```

Или установите вручную:
```bash
pip install ultralytics opencv-python numpy torch torchvision
```

### 2. Загрузите модели YOLO

Модели будут загружены автоматически при первом запуске, но вы также можете скачать их заранее:

**YOLOv8 Segmentation модели:**
- `yolov8n-seg.pt` - Nano (самая быстрая, ~7MB)
- `yolov8s-seg.pt` - Small (~23MB)
- `yolov8m-seg.pt` - Medium (~52MB)
- `yolov8l-seg.pt` - Large (~83MB)
- `yolov8x-seg.pt` - XLarge (самая точная, ~133MB)

Скачать можно с официального репозитория Ultralytics:
- https://github.com/ultralytics/assets/releases

Или модели загрузятся автоматически при первом запуске через библиотеку `ultralytics`.

**Рекомендации:**
- Для работы в реальном времени на CPU используйте `yolov8n-seg.pt`
- Для GPU можно использовать `yolov8s-seg.pt` или `yolov8m-seg.pt`
- Для максимальной точности используйте `yolov8x-seg.pt` (требуется мощный GPU)

## Использование

### 1. Простой просмотр веб-камеры

```bash
python webcam_simple.py
```

Показывает видео с веб-камеры без обработки.

### 2. YOLO сегментация (базовая версия)

```bash
python yolo_seg_webcam.py
```

Запускает YOLO сегментацию с моделью по умолчанию (`yolov8n-seg.pt`).

**Автоматически выбирает GPU если доступен, иначе использует CPU.**

Чтобы использовать другую модель, измените переменную `model_path` в файле.

### 3. YOLO сегментация (расширенная версия)

```bash
python yolo_seg_advanced.py
```

**Параметры командной строки:**

```bash
# Использовать другую модель
python yolo_seg_advanced.py --model yolov8s-seg.pt

# Выбрать устройство (auto/cpu/cuda/gpu)
python yolo_seg_advanced.py --device cuda  # Использовать GPU
python yolo_seg_advanced.py --device cpu   # Использовать CPU
python yolo_seg_advanced.py --device auto  # Автоматический выбор (по умолчанию)

# Изменить порог уверенности
python yolo_seg_advanced.py --conf 0.5

# Изменить порог IOU
python yolo_seg_advanced.py --iou 0.5

# Использовать другую камеру
python yolo_seg_advanced.py --camera 1

# Изменить разрешение
python yolo_seg_advanced.py --width 640 --height 480

# Сохранить видео с результатами
python yolo_seg_advanced.py --save

# Комбинация параметров
python yolo_seg_advanced.py --model yolov8m-seg.pt --device cuda --conf 0.4 --save
```

**Управление во время работы:**
- `q` - выход
- `s` - сделать скриншот
- `+` или `=` - увеличить порог уверенности
- `-` - уменьшить порог уверенности

## Структура файлов

```
yolo-experiments/
├── webcam_simple.py          # Простой просмотр веб-камеры
├── yolo_seg_webcam.py        # YOLO сегментация (базовая)
├── yolo_seg_advanced.py      # YOLO сегментация (расширенная)
├── requirements.txt          # Зависимости
├── README.md                 # Документация
└── yolov8n-seg.pt           # Модель (загружается автоматически)
```

## Системные требования

### Минимальные:
- Python 3.8+
- 4GB RAM
- Веб-камера
- CPU (для yolov8n-seg.pt можно получить ~15-20 FPS)

### Рекомендуемые:
- Python 3.10+
- 8GB+ RAM
- NVIDIA GPU с CUDA (для высокого FPS)
- Веб-камера HD

## Устранение неполадок

### Камера не открывается
- Проверьте, что камера подключена и работает
- Попробуйте изменить индекс камеры: `--camera 1` или `--camera 2`
- Закройте другие приложения, использующие камеру

### Низкий FPS
- Используйте более легкую модель: `yolov8n-seg.pt`
- Уменьшите разрешение: `--width 640 --height 480`
- Убедитесь, что используется GPU (если доступен)

### Ошибка импорта torch
- Установите PyTorch с официального сайта: https://pytorch.org/
- Для GPU установите версию с CUDA:
  ```bash
  # Для Windows с CUDA
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  
  # Для CPU only
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

### GPU не используется
- Проверьте установлен ли CUDA: `nvidia-smi`
- Проверьте доступность в PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`
- Явно укажите устройство: `--device cuda`

### Модель не загружается
- Проверьте подключение к интернету (при первом запуске)
- Скачайте модель вручную и укажите путь

## Дополнительная информация

- Документация Ultralytics YOLO: https://docs.ultralytics.com/
- YOLOv8 GitHub: https://github.com/ultralytics/ultralytics
- OpenCV документация: https://docs.opencv.org/

## Классы объектов

YOLO сегментация обучена на датасете COCO и может обнаруживать 80 классов объектов, включая:
- Люди (person)
- Транспорт (car, bus, truck, bicycle, motorcycle, etc.)
- Животные (dog, cat, bird, horse, etc.)
- Предметы интерьера (chair, table, sofa, bed, etc.)
- Электроника (tv, laptop, keyboard, mouse, etc.)
- И многое другое

Полный список классов можно найти в документации COCO dataset.

