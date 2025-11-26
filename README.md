# 📷 CCTV Attendance System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv&logoColor=white)
![Face Recognition](https://img.shields.io/badge/AI-Face%20Recognition-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> **An AI-powered, real-time face recognition attendance system that automatically marks attendance using CCTV or webcam feeds.**

---

## 🌟 Features

- **⚡ Real-Time Detection**: Instantly detects and recognizes faces from live video feeds.
- **🧠 Advanced AI**: Uses HOG (Histogram of Oriented Gradients) and ResNet deep learning models for 99.38% accuracy.
- **📊 Automated Logging**: Automatically records attendance with Name, Date, Time, and Confidence scores in CSV/Excel format.
- **🛡️ Anti-Spoofing**: High-confidence thresholding to prevent false positives.
- **🖥️ Modern GUI**: User-friendly interface with live statistics, attendance logs, and camera controls.
- **📈 Smart Analytics**: Tracks "Present" vs "Absent" status and visualizes attendance data.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| **Core** | Python 3.x | Main programming language |
| **Vision** | OpenCV (`cv2`) | Image processing and video capture |
| **AI/ML** | `dlib` & `face_recognition` | Facial landmark detection and encoding |
| **Data** | Pandas & NumPy | Data manipulation and CSV logging |
| **GUI** | Tkinter | Desktop application interface |

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CMake (required for dlib)

### 1. Clone the Repository
```bash
git clone https://github.com/abhishekk-y/CCTV_Attendance_System.git
cd CCTV_Attendance_System
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*Note: Installing `dlib` may take a few minutes. If you encounter issues, ensure CMake is installed.*

---

## 📖 Usage

### 1. Register New Faces
To add people to the database:
1. Run the application:
   ```bash
   python attendance_gui.py
   ```
2. Click **"➕ Add New Face"**.
3. Enter the person's name.
4. Look at the camera and click **"Capture"**.
5. Click **"Save"** to store the face encoding.

Alternatively, you can manually add images to the `known_faces/` folder (e.g., `john_doe.jpg`). Then run:
```bash
python encode_faces.py
```

### 2. Start Attendance System
Run the main GUI application:
```bash
python attendance_gui.py
```
- The system will start the camera feed.
- Detected faces will be marked with a green box and name.
- Attendance is automatically saved to `attendance1.csv`.

### 3. View Records
Open `attendance1.csv` to view the logs:
```csv
Name,Date,Time,Confidence
John Doe,2023-10-25,09:00:01,98.5
Jane Smith,2023-10-25,09:05:22,99.1
```

---

## 📂 Project Structure

```
CCTV_Attendance_System/
├── AI_Models/                # Pre-trained dlib models
├── known_faces/              # Database of registered user images
├── attendance1.csv           # Daily attendance log file
├── attendance_gui.py         # 🖥️ Main GUI Application
├── capture_image.py          # Script to capture training images
├── encode_faces.py           # Script to generate face encodings
├── encodings.pickle          # Serialized face data (generated)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ Configuration

You can adjust system parameters in `attendance_gui.py`:

```python
# Performance Tuning
self.process_every_n_frames = 4  # Process every 4th frame (Higher = Faster, Lower = More Accurate)
tolerance = 0.6                  # Match strictness (Lower = Stricter)
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built by  <a href="https://github.com/abhishekk-y">Abhishek</a></sub>
</div>
