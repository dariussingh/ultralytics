from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(data="./datasets/coco8.yaml", epochs=1)

model.val(data="./datasets/coco8.yaml")

model.predict("./ultralytics/assets/bus.jpg")
