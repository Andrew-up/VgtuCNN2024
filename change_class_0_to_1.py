import os


# Укажите путь к папке с файлами меток
PATH_DATASET = r"D:\myProgramm\VgtuCNN2024\valid_dataset_helicopter_and_drone\test\labels"


def update_class_in_labels(labels_folder, old_class, new_class):
    for filename in os.listdir(labels_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(labels_folder, filename)

            with open(file_path, 'r') as file:
                lines = file.readlines()

            updated_lines = []
            for line in lines:
                parts = line.split()
                if parts and parts[0] == str(old_class):
                    parts[0] = str(new_class)
                updated_lines.append(" ".join(parts) + "\n")

            with open(file_path, 'w') as file:
                file.writelines(updated_lines)
            print(f"Updated: {file_path}")


# Заменяем класс 0 на 1
update_class_in_labels(PATH_DATASET, old_class=0, new_class=1)
