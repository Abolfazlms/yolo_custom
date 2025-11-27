📦 YOLOv3 Real-Time Object Detection (OpenCV + Python)








A real-time object detection system using YOLOv3, OpenCV, and a standard webcam.
Each detected object is drawn with a unique, consistent class-based color, and labels include confidence percentages.

Perfect for beginners exploring computer vision, students presenting AI demos, or developers building real-time recognition systems.

🚀 Features

✔️ Real-time detection via webcam

✔️ YOLOv3 + OpenCV DNN module (no GPU required)

✔️ Unique & persistent color for every object class

✔️ Confidence thresholding + Non-Maximum Suppression

✔️ User-friendly visualization with bounding boxes and labels

✔️ Clean modular code (find_objects() for detection logic)

📂 Project Structure
.
├── yolo_object_detection.py
├── yolov3.cfg
├── yolov3.weights
├── coco.names
└── README.md

📥 Download YOLOv3 Weights

Download the official YOLOv3 weights (required):

🔗 https://pjreddie.com/media/files/yolov3.weights

(or use your Google Drive link if preferred)

Place them in the same folder as the script.

🛠️ Installation
1️⃣ Install dependencies
pip install opencv-python numpy

2️⃣ Run the script
python yolo_object_detection.py

🎮 Controls
Key	Action
q	Quit the webcam and close program
🧠 How It Works

The webcam frame is converted into a YOLO input blob.

The network performs a forward pass to get predictions.

For each detection:

The class ID is found.

Confidence is checked.

Bounding box is scaled to the actual image.

Non-Maximum Suppression (NMS) removes overlapping detections.

Each class is drawn using a unique, random but stable BGR color.

🖼️ Output Example

Real-time bounding boxes look like this:

[PERSON 97%]
[CHAIR 84%]
[DOG 91%]


Each category has its own color for easy visualization.

📌 Code Highlights
Unique color per class
colors = np.random.uniform(50, 255, size=(len(coco_classes), 3))

Drawing boxes
cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

Applying NMS
indices = cv2.dnn.NMSBoxes(bbox, confidences, confidence_threshold, nms_threshold)

📌 Future Improvements

🔹 Add FPS counter

🔹 Enable GPU acceleration (CUDA build of OpenCV)

🔹 Add object tracking (SORT / DeepSORT)

🔹 Save detections to file

🔹 Run on video files instead of webcam

📜 License

MIT License — free to use, modify, and distribute.