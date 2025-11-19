import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

from ultralytics import YOLO

# DATASET_PATH = "D:\myProgramm\datasets\Cats.v3-augmented-v1-accurate.yolov8\data.yaml"
# DATASET_PATH = "D:\myProgramm\datasets\Chicken Detection and Tracking.v12-raw-images-roosterlabel.yolov8\data.yaml"
# DATASET_PATH = r"D:\myProgramm\datasets\fixed wing UAV dataset.v1-fixed-wing-uav.yolov8\data.yaml"
# DATASET_PATH = r"D:\myProgramm\datasets\number.v1i.yolov8\data.yaml"
DATASET_PATH = r"D:\myProgramm\datasets\Object detection.v10-900-images-ka-version.yolov8\data.yaml"


@dataclass
class YOLOModelInfo:
    version: Optional[str] = None
    model_date: Optional[str] = None
    model_type: Optional[str] = None
    framework: Optional[str] = None
    format: Optional[str] = None
    classes: Optional[List[str]] = None
    num_classes: Optional[int] = None
    input_size: Optional[int] = None
    confidence_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None
    max_detections: Optional[int] = None
    map50: Optional[float] = None
    map50_95: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    inference_speed: Optional[float] = None
    training_epochs: Optional[int] = None
    dataset_size: Optional[int] = None
    augmentation: Optional[bool] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    dataset: Optional[str] = None
    author: Optional[str] = None
    description: Optional[str] = None
    license: Optional[str] = None
    checksum: Optional[str] = None


def main(dataset_path):
    # Load a model
    model = YOLO("yolo11n.yaml")

    # Train the model
    results = model.train(data=dataset_path, epochs=50, imgsz=640,
                          batch=0.55, patience=35)

    # Сохраняем информацию в файл
    save_dir = os.path.dirname(model.trainer.best) if hasattr(model.trainer, 'best') else model.trainer.save_dir

    # Создаем и сохраняем информацию о модели
    model_info = create_and_save_model_info(model, results, save_dir)

    return results, model


def create_and_save_model_info(model, results, save_dir: str):
    """Создает экземпляр YOLOModelInfo и сохраняет в файл"""

    # Создаем экземпляр с автоматически заполняемыми полями
    model_info = YOLOModelInfo()

    # Автоматически заполняемые поля из модели и результатов
    if hasattr(model, 'model') and hasattr(model.model, 'nc'):
        model_info.num_classes = model.model.nc

    if hasattr(model, 'names'):
        model_info.classes = list(model.names.values())

    if hasattr(model.trainer, 'args'):
        args = model.trainer.args
        model_info.input_size = getattr(args, 'imgsz', 640)
        model_info.training_epochs = getattr(args, 'epochs', 50)
        model_info.batch_size = getattr(args, 'batch', 16)
        model_info.learning_rate = getattr(args, 'lr0', 0.01)
        model_info.dataset = getattr(args, 'data', 'unknown')

    # Метрики из результатов обучения
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        model_info.map50 = metrics.get('metrics/mAP50(B)')
        model_info.map50_95 = metrics.get('metrics/mAP50-95(B)')
        model_info.precision = metrics.get('metrics/precision(B)')
        model_info.recall = metrics.get('metrics/recall(B)')

    # Поля со значениями по умолчанию
    model_info.version = "1.0.0"
    model_info.model_date = datetime.now().strftime("%Y-%m-%d")
    model_info.model_type = "YOLOv11n"
    model_info.framework = "YOLOv11"
    model_info.format = "pt"
    model_info.author = "Unknown"

    # Сохраняем в файл
    save_model_info_to_txt(model_info, save_dir)

    return model_info


def save_model_info_to_txt(model_info: YOLOModelInfo, save_dir: str):
    """Сохраняет YOLOModelInfo в формате key=value"""

    info_file_path = os.path.join(save_dir, "info.txt")

    with open(info_file_path, 'w', encoding='utf-8') as f:
        # Проходим по всем полям датакласса
        for field_name in model_info.__dataclass_fields__:
            value = getattr(model_info, field_name)

            # Форматируем значение в зависимости от типа
            if value is None:
                formatted_value = "null"
            elif isinstance(value, list):
                formatted_value = ",".join(map(str, value))
            elif isinstance(value, bool):
                formatted_value = str(value).lower()
            else:
                formatted_value = str(value)

            f.write(f"{field_name}={formatted_value}\n")

    print(f"Model info saved to: {info_file_path}")


if __name__ == '__main__':
    main(DATASET_PATH)
