# pip install ultralytics
from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt

# 1. Load pre-trained YOLO model
model = YOLO('yolov8s.pt')  # small pre-trained YOLOv8 model

# 2. Load input image
img_path = 'sample_image.jpg'
results = model(img_path)

# 3. Print detected objects
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls)
        conf = box.conf.item()
        print(f"Detected: {model.names[cls]} (Confidence: {conf:.2f})")

# 4. Display result image with bounding boxes
res_img = results[0].plot()
plt.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Object Detection using YOLOv8")
plt.show()
