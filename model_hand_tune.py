import os
import time

import cv2
import pandas as pd
from ultralytics import YOLO

MODEL_PATH = r"D:\Test\Test1\model\runs\Best.pt"
TEST_IMAGES = r"D:\D:\Test\Test1\dataset_test_images\test\images"
DATASET_PATH_TRAIN = r"D:\Test\Test1\dataset_train\data.yaml"
DATASET_PATH_VALID = r"D:\Test\Test1\dataset_valid\data.yaml"

freeze = 10
batch_size = 4
lr = 1e-5
imgsize = 640
dropout = 0.2
epoch = 150


def model_hand_tune(model_path, test_path_images, dataset_train, dataset_valid):
    image_files = sorted(
        [file for file in os.listdir(test_path_images) if file.lower().endswith(('.png', '.jpg', '.jpeg'))])
    image_files = image_files[:200]
    images = []
    # Загружаем и преобразуем изображения в Numpy массив
    for file in image_files:
        image_path = os.path.join(test_path_images, file)
        image = cv2.imread(image_path)  # Загружаем изображение
        if image is not None:
            images.append(image)  # Добавляем изображение в список

    model = YOLO(model_path)  # load a pretrained model (recommended for training)

    model_name = f'model_{imgsize}_image'
    results = model.train(data=dataset_train,
                          epochs=epoch,
                          exist_ok=True,
                          imgsz=imgsize,
                          patience=35,
                          project=r'D:\myProgramm\VgtuCNN2024\model_temp_train',
                          name=model_name,
                          batch=8,
                          )

    print("freeze:", freeze)
    print("dropout:", dropout)
    print("lr:", lr)
    val_model = YOLO(rf"D:\myProgramm\VgtuCNN2024\model_temp_train\{model_name}\weights\best.pt")
    # Получаем количество параметров
    time_start = time.time()
    for i in images:
        val_model.predict(i)
    time_end = time.time()
    print('FPS:', len(image_files) / (time_end - time_start))
    fps = len(image_files) / (time_end - time_start)
    res = val_model.val(data=dataset_valid, save=False)

    data_all = {"epochs": epoch,
                "freeze": freeze,
                "batch_size": batch_size,
                "lr": lr,
                "imgsize": imgsize,
                "mAP50-95": res.box.map,
                "mAP50": res.box.map50,
                "mAP75": res.box.map75,
                "dropout": dropout,
                'map50_drone': res.box.ap50[0],
                'map50_helicopter': res.box.ap50[1],
                'map50-95_drone': res.box.ap[0],
                'map50-95_helicopter': res.box.ap[1],
                'FPS': fps}
    merge_data = [data_all]

    try:
        # Сохранение в Excel
        output_path = r"D:\myProgramm\VgtuCNN2024\validation_results_2_image_helicopter.xlsx"
        if not os.path.exists(output_path):
            df = pd.DataFrame(
                columns=["epochs", "freeze", "batch_size", "lr", "imgsize", "mAP50-95", "mAP50", "mAP75", "dropout",
                         "map50_drone", "map50_helicopter", "map50-95_drone", "map50-95_helicopter", "FPS"])
            df.to_excel(output_path, index=False)
            print(f"Файл '{output_path}' создан.")

        new_df = pd.DataFrame(merge_data)

        # Добавление данных в существующий Excel файл
        with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            new_df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
        print(f"Результаты сохранены в файл: {output_path}")
    except Exception as e:
        print(e)


if __name__ == '__main__':
    model_hand_tune(MODEL_PATH, TEST_IMAGES, DATASET_PATH_TRAIN, DATASET_PATH_VALID)
