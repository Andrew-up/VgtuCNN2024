from ultralytics import YOLO

MODEL_PATH = r"D:\Test\Test1\model\runs\Best.pt"
DATASET_PATH_TRAIN = r"D:\Test\Test1\dataset_train\data.yaml"

def model_auto_tune(model_yolo_path, model_dataset_path):
    model = YOLO(model_yolo_path)
    epoch = 50
    imgsize = 640
    res_train = model.tune(data=model_dataset_path,
                           epochs=epoch,
                           iterations=250,
                           exist_ok=True,
                           batch=8)

if __name__ == '__main__':
    model_auto_tune(MODEL_PATH, DATASET_PATH_TRAIN)
