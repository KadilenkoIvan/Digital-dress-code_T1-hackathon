"""
Экспорт YOLO11 модели в ONNX с квантованием
"""
from ultralytics import YOLO
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolo11n-seg.pt', help='Модель для экспорта')
    parser.add_argument('--imgsz', type=int, default=640, help='Размер входа')
    parser.add_argument('--half', action='store_true', help='FP16 (для GPU, быстрее)')
    parser.add_argument('--int8', action='store_true', help='INT8 (только для OpenVINO, очень быстро на CPU)')
    parser.add_argument('--format', type=str, default='onnx', 
                        choices=['onnx', 'engine', 'openvino'],
                        help='Формат экспорта: onnx, engine (TensorRT), openvino')
    args = parser.parse_args()
    
    print("="*70)
    print(f"Экспорт {args.model}")
    print(f"Формат: {args.format.upper()}")
    print(f"Размер: {args.imgsz}")
    print(f"FP16: {args.half}")
    print(f"INT8: {args.int8}")
    print("="*70)
    
    # Проверка совместимости INT8
    if args.int8 and args.format != 'openvino':
        print("\n⚠️ INT8 поддерживается только для OpenVINO!")
        print("Автоматически переключаю на --format openvino")
        args.format = 'openvino'
    
    model = YOLO(args.model)
    
    # Параметры экспорта
    export_kwargs = {
        'format': args.format,
        'imgsz': args.imgsz,
        'half': args.half,
    }
    
    # Специфичные параметры для ONNX
    if args.format == 'onnx':
        export_kwargs['simplify'] = True
        export_kwargs['dynamic'] = False
        export_kwargs['opset'] = 12
    
    # TensorRT параметры
    elif args.format == 'engine':
        if not args.half:
            print("⚠️ Для TensorRT рекомендуется --half для лучшей производительности")
        export_kwargs['workspace'] = 4  # GB
    
    # OpenVINO параметры
    elif args.format == 'openvino':
        export_kwargs['int8'] = args.int8
        if args.int8:
            print("OpenVINO с INT8 квантованием (макс. скорость на CPU)")
        else:
            print("OpenVINO оптимизирован для Intel CPU/GPU")
    
    path = model.export(**export_kwargs)
    
    suffix = f" ({args.format.upper()}"
    if args.half:
        suffix += " FP16)"
    elif args.int8:
        suffix += " INT8)"
    else:
        suffix += ")"
    
    print(f"\n✅ Экспортировано: {path}{suffix}")
    
    # Рекомендации по запуску
    print(f"\nЗапуск:")
    if args.format == 'onnx':
        if args.half:
            print(f"  python yolo_seg_onnx.py --model {path} --device cuda")
        else:
            print(f"  python yolo_seg_onnx.py --model {path} --device cpu")
            print(f"  python yolo_seg_onnx.py --model {path} --device cuda")
    elif args.format == 'engine':
        print(f"  # TensorRT работает только на GPU")
        print(f"  python yolo_seg_onnx.py --model {path} --device cuda")
    elif args.format == 'openvino':
        print(f"  # OpenVINO оптимизирован для Intel CPU")
        print(f"  python yolo_seg_onnx.py --model {path} --device cpu")
    
    print(f"\n💡 Квантование ({suffix}) уже в модели, дополнительных параметров не нужно!")
    
    # Дополнительные рекомендации
    print(f"\nСравнение форматов:")
    print(f"  ONNX:         универсальный, CPU/GPU")
    print(f"  TensorRT:     макс. скорость на NVIDIA GPU")
    print(f"  OpenVINO:     макс. скорость на Intel CPU")
    print(f"  OpenVINO INT8: ~2-4x быстрее на CPU")
    
    print(f"\nПримеры экспорта:")
    print(f"  # ONNX FP16 для GPU")
    print(f"  python export_to_onnx.py --model {args.model} --format onnx --half")
    print(f"  # TensorRT FP16 для NVIDIA GPU")
    print(f"  python export_to_onnx.py --model {args.model} --format engine --half")
    print(f"  # OpenVINO INT8 для CPU (максимальная скорость)")
    print(f"  python export_to_onnx.py --model {args.model} --format openvino --int8")


if __name__ == "__main__":
    main()

