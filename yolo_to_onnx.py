
from ultralytics import YOLO

def main():
    # model = YOLO('best.pt')
    model = YOLO(r'D:\myProgramm\VgtuCNN2024\runs\detect\train16\weights\best.pt')
    model.export(format='onnx')



if __name__ == '__main__':
    main()