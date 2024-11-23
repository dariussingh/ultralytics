from ultralytics import YOLO

model = YOLO(".\\ultralytics\\cfg\\models\\11\\yolo11n-mad.yaml")

model.train(data=".\\ultralytics\\cfg\\datasets\\coco-mad.yaml", epochs=1)

# model.val(data=".\\ultralytics\\cfg\\datasets\\coco-mad.yaml")

# model.predict("./ultralytics/assets/bus.jpg")
