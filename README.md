# 📦 YOLOv3 Real-Time Object Detection (OpenCV + Python)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-dnn-red)
![YOLO](https://img.shields.io/badge/YOLO-v3-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

A **real-time object detection system** using **YOLOv3**, **OpenCV**, and a standard webcam.  
Each detected object is drawn with a **unique, consistent class-based color**, and labels include confidence percentages.

Perfect for beginners exploring computer vision, students presenting AI demos, or developers building real-time recognition systems.

---

## 🚀 Features

- ✔️ Real-time detection via webcam  
- ✔️ YOLOv3 + OpenCV DNN module (no GPU required)  
- ✔️ Unique & persistent color for every object class  
- ✔️ Confidence thresholding + Non-Maximum Suppression  
- ✔️ Modular & clean detection function (`find_objects()`)  
- ✔️ High-quality bounding boxes and labels  

---

## 📂 Project Structure

.
├── yolo_object_detection.py
├── yolov3.cfg
├── yolov3.weights
├── coco.names
└── README.md

yaml
Copy code

---

## 📥 Download YOLOv3 Weights

You must download the official YOLOv3 weights:

🔗 [Download YOLOv3 weights](https://pjreddie.com/media/files/yolov3.weights)  
(or use the provided Google Drive link)

Place the file **yolov3.weights** in the project folder.

---

## 🛠️ Installation

### 1️⃣ Install dependencies

```bash
pip install opencv-python numpy
```

2️⃣ Run the script
```bash
python yolo_object_detection.py
```
## 🎮 Controls

| Key  | Action                          |
|------|---------------------------------|
| **q** | Quit the webcam and close the program |

---

## 🧠 How It Works

1. The webcam frame is converted to a YOLO input **blob**.  
2. The YOLO network performs a forward pass and outputs detections.  
3. For each detection:
   - Class ID is extracted  
   - Confidence score is checked  
   - Bounding box is scaled back to the original frame size  
4. **Non-Maximum Suppression (NMS)** removes overlapping predictions.  
5. Unique, stable colors are assigned per class and drawn on the frame.

---

## 🖼️ Output Example

You may see labels like:

PERSON 97%
DOG 91%
CHAIR 84%

Each category is displayed with a unique color.

---

## 📌 Important Code Snippets

### Unique stable colors for each class

```python
colors = np.random.uniform(50, 255, size=(len(coco_classes), 3))
Drawing bounding boxes
python
Copy code
cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
cv2.putText(image, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
Applying Non-Maximum Suppression
python
Copy code
indices = cv2.dnn.NMSBoxes(
    bbox, confidences, confidence_threshold, nms_threshold
)
```
---
## 🔧 Future Improvements

- Add FPS counter  
- GPU support (CUDA build of OpenCV)  
- Save detection results  
- Add object tracking (DeepSORT / SORT)  
- Video file input support  
- GUI interface  

---

## 📜 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.
