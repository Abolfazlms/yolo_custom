import cv2
import numpy as np
# download weights of algorithm from :
#https://drive.google.com/file/d/10_C4Eu-xtBOnlQvxdCQtURGG3fAAxP8D/view?usp=sharing

# Initialize webcam
video = cv2.VideoCapture(0)

# Paths to YOLO files
coco_file = "coco.names"
net_config = "yolov3.cfg"
net_weight = "yolov3.weights"

# YOLO parameters
blob_size = 320
confidence_threshold = 0.5
nms_threshold = 0.3

# Load COCO class names
with open(coco_file, "rt") as f:
    coco_classes = [line.strip() for line in f.readlines()]

# Generate unique and consistent random colors for each class (BGR format)
colors = np.random.uniform(50, 255, size=(len(coco_classes), 3))

# Load YOLOv3 network
net = cv2.dnn.readNetFromDarknet(net_config, net_weight)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def find_objects(outputs, image):
    """
    Process YOLO output layers and draw bounding boxes with class-specific colors
    """
    hT, wT, cT = image.shape
    bbox = []
    class_ids = []
    confidences = []

    # Iterate through all output layers
    for output in outputs:
        for detection in output:
            scores = detection[5:]                    # Class probabilities (80 classes)
            class_id = np.argmax(scores)              # Index of predicted class
            confidence = scores[class_id]             # Confidence score

            # Filter detections by confidence threshold
            if confidence > confidence_threshold:
                # Scale bounding box coordinates to original image size
                w = int(detection[2] * wT)
                h = int(detection[3] * hT)
                x = int((detection[0] * wT) - w / 2)
                y = int((detection[1] * hT) - h / 2)

                bbox.append([x, y, w, h])
                class_ids.append(class_id)
                confidences.append(float(confidence))

    # Apply Non-Maximum Suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(bbox, confidences, confidence_threshold, nms_threshold)

    # Draw bounding boxes and labels
    if len(indices) > 0:
        for i in indices.flatten():  # Flatten to get correct indices
            box = bbox[i]
            x, y, w, h = box

            # Get unique color for this class
            color = colors[class_ids[i]]
            color = (int(color[0]), int(color[1]), int(color[2]))  # Convert to integer tuple

            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            # Prepare label: CLASS_NAME CONFIDENCE%
            label = f"{coco_classes[class_ids[i]].upper()} {int(confidences[i] * 100)}%"

            # Draw label text above the box
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# Main loop
while True:
    success, frame = video.read()
    if not success:
        print("Failed to grab frame")
        break

    # Create blob from image (preprocessing for YOLO)
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1/255.0, size=(blob_size, blob_size),
        mean=(0, 0, 0), swapRB=True, crop=False
    )

    # Set input to the network
    net.setInput(blob)

    # Get output layer names and perform forward pass
    output_layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layer_names)

    # Detect objects and draw results
    find_objects(outputs, frame)

    # Display the result
    cv2.imshow("YOLO Object Detection - Unique Color per Class", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video.release()
cv2.destroyAllWindows()