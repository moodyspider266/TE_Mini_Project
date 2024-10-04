import cv2
import torch
import numpy as np
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
from utils.torch_utils import select_device

# Set your model, video, and device
weights = 'runs/train/gelan-c-det/weights/best.pt'  # Trained model weights
source = 'test_3.mp4'   # Path to the video file
img_size = 640                            # Image size
device = "cpu"              # GPU or CPU

# Load the model
model = DetectMultiBackend(weights, device=device)
stride, names, pt = model.stride, model.names, model.pt
img_size = img_size

# Class-specific confidence thresholds
class_conf_thresholds = {
    'electric-pole': 0.6,
    'hoarding': 0.8,
    'street-vendor': 0.5,
    'background': 0.4,
}

# Define colors for each class (in BGR format for OpenCV)
class_colors = {
    'electric-pole': (0, 0, 255),   # Red
    'hoarding': (255, 0, 0),        # Blue
    'street-vendor': (0, 255, 0),   # Green
    'background': (0, 255, 255),    # Yellow
}

# Load the video
cap = cv2.VideoCapture(source)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and pad image while maintaining aspect ratio
    img = letterbox(frame, img_size, stride=stride)[0]

    # Convert frame to RGB and transpose for PyTorch (from HWC to CHW)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)

    # Convert image to Tensor and send to device
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0  # Normalize to [0,1]
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Run inference
    pred = model(img)
    pred = non_max_suppression(pred, 0.25, 0.45)  # Initial NMS with global thresholds

    # Process the detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to original frame size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in det:
                class_name = names[int(cls)]

                # Apply class-specific confidence thresholds
                if conf >= class_conf_thresholds.get(class_name, 0.25):  # Default to 0.25 if class not found
                    label = f'{class_name} {conf:.2f}'
                    print(f"Detected {label}")
                    
                    # Get the color for the current class
                    color = class_colors.get(class_name, (0, 255, 0))  # Default to green if class not found

                    # Draw the detection on the frame with the class-specific color
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    out.write(frame)  # Write the frame with detections to the output video

# Release video and close everything
cap.release()
out.release()
