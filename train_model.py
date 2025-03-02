from ultralytics import YOLO

DATASET_PATH = "D:\Test\Test1\DRONES_NEW.v4i.yolov11\data.yaml"


def main(dataset_path):
    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    # Train the model with 2 GPUs
    results = model.train(data=dataset_path, epochs=250, imgsz=640,
                          batch=0.55, patience=35)


if __name__ == '__main__':
    main(DATASET_PATH)
