import os
import time

import cv2
import pandas as pd
from ultralytics import YOLO

"""
 Этот файл постоянно редактируется, тут много чего менялось, но смысл надеюсь получится понять :)
"""


def main():
    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    # Train the model with 2 GPUs
    results = model.train(data="D:\myProgramm\VgtuCNN2024\DRONES_NEW.v4i.yolov11\data.yaml", epochs=250, imgsz=640,
                          batch=0.55, patience=35)


def model_testing():
    model = YOLO(r"D:\myProgramm\VgtuCNN2024\res_model\test_model\weights\best.pt")
    # model = YOLO(r"D:\myProgramm\VgtuCNN2024\runs\detect\train31\weights\best.pt")
    # model.val(data=r'D:\myProgramm\VgtuCNN2024\dataset\data.yaml')
    # res = model.predict(r'D:\myProgramm\VgtuCNN2024\dataset\test\images\cats25_jpg.rf.37b9655d22612f070bd505b090d7ac9a.jpg')
    # res[0].show()
    res = model.predict(r'D:\myProgramm\VgtuCNN2024\dataset_merge\arkin-si-nkIIbgOVyl4.jpg', conf=0.55)
    res[0].show()
    # model.val(data=r'D:\myProgramm\VgtuCNN2024\dataset_airplane\data.yaml')


def model_tune(model_yolo_path, model_dataset_path, output_path):
    model = YOLO(model_yolo_path)
    epoch = 50
    imgsize = 640
    res_train = model.tune(data=model_dataset_path,
                           epochs=epoch,
                           iterations=250,
                           exist_ok=True,
                           project=output_path,
                           batch=8)

    exit(0)
    # res_train.
    val_model = YOLO(rf"D:\myProgramm\VgtuCNN2024\model_temp_train\tune\weights\best.pt")
    res_val = val_model.val(data=r'D:\myProgramm\VgtuCNN2024\valid_dataset_helicopter_and_drone\data.yaml', save=False)
    data_all = {"epochs": epoch,
                "imgsize": imgsize,
                "mAP50-95": res_val.box.map,
                "mAP50": res_val.box.map50,
                "mAP75": res_val.box.map75,
                'map50_drone': res_val.box.ap50[0],
                'map50-95_drone': res_val.box.ap[0],
                'model_aug': 'tune'}

    merge_data = [data_all]
    try:
        # Сохранение в Excel
        output_path = rf"D:\myProgramm\VgtuCNN2024\model_tune.xlsx"
        if not os.path.exists(output_path):
            df = pd.DataFrame(
                columns=["epochs", "imgsize", "mAP50-95", "mAP50", "mAP75", 'map50_drone', 'map50-95_drone',
                         'model_aug'])
            df.to_excel(output_path, index=False)
            print(f"Файл '{output_path}' создан.")
        # df.to_excel(output_path, index=False)
        # Добавление новой строки в DataFrame
        new_df = pd.DataFrame(merge_data)

        # Добавление данных в существующий Excel файл
        with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            new_df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
        print(f"Результаты сохранены в файл: {output_path}")
    except Exception as e:
        print(e)


def train_augmentation(yaml_file_path: str | None, model_name='test_123'):
    model = YOLO(r"best2.pt")
    epoch = 50
    imgsize = 640
    res_train = model.train(data="D:\myProgramm\VgtuCNN2024\dataset_helicopter_and_drone\data.yaml",
                            cfg=yaml_file_path,
                            epochs=epoch,
                            exist_ok=True,
                            imgsz=imgsize,
                            patience=35,
                            project=r'D:\myProgramm\VgtuCNN2024\model_temp_train',
                            name=model_name,
                            freeze=10,
                            batch=8)

    # res_train.
    val_model = YOLO(rf"D:\myProgramm\VgtuCNN2024\model_temp_train\{model_name}\weights\best.pt")
    res_val = val_model.val(data=r'D:\myProgramm\VgtuCNN2024\DRONES_NEW.v4i.yolov11\data.yaml', save=False)
    data_all = {"epochs": epoch,
                "imgsize": imgsize,
                "mAP50-95": res_val.box.map,
                "mAP50": res_val.box.map50,
                "mAP75": res_val.box.map75,
                'map50_drone': res_val.box.ap50[0],
                'map50-95_drone': res_val.box.ap[0],
                'model_aug': model_name}

    merge_data = [data_all]
    try:
        # Сохранение в Excel
        output_path = rf"D:\myProgramm\VgtuCNN2024\validation_results_augmentstion.xlsx"
        if not os.path.exists(output_path):
            df = pd.DataFrame(
                columns=["epochs", "imgsize", "mAP50-95", "mAP50", "mAP75", 'map50_drone', 'map50-95_drone',
                         'model_aug'])
            df.to_excel(output_path, index=False)
            print(f"Файл '{output_path}' создан.")
        # df.to_excel(output_path, index=False)
        # Добавление новой строки в DataFrame
        new_df = pd.DataFrame(merge_data)

        # Добавление данных в существующий Excel файл
        with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            new_df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
        print(f"Результаты сохранены в файл: {output_path}")
    except Exception as e:
        print(e)


def model_few_shot_tuning():
    lr_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    batch_size_list = [2, 4, 6, 8, 10]
    random_img_size = [920, 1024, 64, 128, 256, 320, 420, 560, 600]
    # random_img_size = sorted([i for i in (random_img_size)])
    print(random_img_size, type(random_img_size))
    # exit(0)
    dropout_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    freeze_list = [15]
    # random.seed(42)
    folder_path = r"D:\myProgramm\VgtuCNN2024\DRONES_NEW.v4i.yolov11_rebalance\test\images"

    image_files = sorted([file for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))])
    image_files = image_files[:200]
    images = []
    # Загружаем и преобразуем изображения в Numpy массив
    for file in image_files:
        image_path = os.path.join(folder_path, file)
        image = cv2.imread(image_path)  # Загружаем изображение
        if image is not None:
            images.append(image)  # Добавляем изображение в список

    # for i in random_img_size:
    model = YOLO("best2.pt")  # load a pretrained model (recommended for training)
    freeze = 10
    # freeze = random.choice(freeze_list)
    # freeze_list.remove(freeze)
    batch_size = 4
    # batch_size_list.remove(batch_size)
    lr = 1e-5
    # lr_list.remove(lr)
    imgsize = 640
    dropout = 0.2
    epoch = 150
    model_name = f'model_{imgsize}_2_image'
    results = model.train(data="D:\myProgramm\VgtuCNN2024\dataset_helicopter_and_drone\data.yaml",
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
    path = r"D:\myProgramm\VgtuCNN2024\res_model\test_model\weights\best.pt"
    # val_model = YOLO(path)
    # Получаем количество параметров
    time_start = time.time()
    for i in images:
        val_model.predict(i)
    time_end = time.time()
    print('FPS:', len(image_files) / (time_end - time_start))
    fps = len(image_files) / (time_end - time_start)
    res = val_model.val(data=r'D:\myProgramm\VgtuCNN2024\valid_dataset_helicopter_and_drone\data.yaml', save=False)

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


def train_vaso():
    # model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    model = YOLO(r'D:\myProgramm\VgtuCNN2024\runs\detect\train6\weights\best.pt')
    model.val()
    # Train the model with 2 GPUs
    # results = model.train(data="D:\myProgramm\VgtuCNN2024\dataset-vaso\data.yaml", epochs=250, imgsz=640)


def model_to_ncnn():
    model = YOLO(r'D:\myProgramm\VgtuCNN2024\model_temp_train\best2_fine_tuning_640_3_image\weights\best.pt')
    model.export(format="ncnn")


def model_train2():
    model = YOLO('best2.pt')
    results = model.train(data="D:\myProgramm\VgtuCNN2024\dataset_helicopter_and_drone\data.yaml",
                          epochs=250,
                          exist_ok=True,
                          imgsz=640,
                          patience=35,
                          project=r'D:\myProgramm\VgtuCNN2024\model_temp_train',
                          name='best2_fine_tuning_640_3_image',
                          batch=4,
                          freeze=10,
                          )


if __name__ == '__main__':
    # model_train2()
    # train_vaso()
    # model_tune(r"D:\myProgramm\VgtuCNN2024\model_temp_train\model_640\weights\best.pt",
    #            "D:\myProgramm\VgtuCNN2024\dataset_helicopter_and_drone\data.yaml",
    #            r'D:\myProgramm\VgtuCNN2024\model_temp_train')
    # exit(0)
    # train_augmentation(model_name='best_hyperparameters', yaml_file_path=r"D:\myProgramm\VgtuCNN2024\aug_yaml\best_hyperparameters.yaml")

    # train_augmentation(model_name='default', yaml_file_path=None)
    # train_augmentation(model_name='hsv_s_degrees_translate_scale',
    #                    yaml_file_path=r'D:\myProgramm\VgtuCNN2024\aug_yaml\hsv_s_degrees_translate_scale.yaml')
    # train_augmentation(model_name='bgr_shear_hsv_v',
    #                    yaml_file_path=r'D:\myProgramm\VgtuCNN2024\aug_yaml\bgr_shear_hsv_v.yaml')
    # train_augmentation(model_name='no_augmentation', yaml_file_path=r"D:\myProgramm\VgtuCNN2024\aug_yaml\no_aug.yaml")
    # train_augmentation(model_name='flip_h_v', yaml_file_path=r"D:\myProgramm\VgtuCNN2024\aug_yaml\flip_h_v.yaml")
    # train_augmentation(model_name='perspective_mosaic',
    #                    yaml_file_path=r"D:\myProgramm\VgtuCNN2024\aug_yaml\perspective_mosaic.yaml")
    # model_few_shot_tuning()
    # model = YOLO('best2.pt')
    # model.val(data = 'D:\myProgramm\VgtuCNN2024\DRONES_NEW.v4i.yolov11_rebalance\data.yaml')
    # model_testing()
    model_to_ncnn()

    # main()
