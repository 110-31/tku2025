from ultralytics import YOLO

# Load a model
model = YOLO("last.pt")

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("aa.jpg")
results[0].show()
