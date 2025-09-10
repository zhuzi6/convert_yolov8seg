from ultralytics import YOLO

model = YOLO("weights/seg_yolo_box_250530.pt")
results = model("20250530_173434_rgb.png")  # Predict on an image
results[0].show()  # Display results
