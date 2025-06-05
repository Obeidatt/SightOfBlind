# SightOfBlind 👁️🧠🎙️

**SightOfBlind** is a real-time assistive system designed to support blind and visually impaired individuals by enabling object-based indoor navigation using voice commands and spoken feedback. It is fully offline, wearable, and powered by edge AI using a Raspberry Pi 5 and YOLOv11 object detection.

---

## 🎯 Project Overview

Visually impaired individuals face constant challenges in navigating unfamiliar environments. SightOfBlind aims to restore autonomy by allowing users to say commands like “Where is the door?” or “Find the chair,” and receive real-time verbal guidance based on AI-powered object recognition and direction awareness.

---

## ⚙️ Key Features

- 🎙️ Natural voice command interface via Speech-to-Text (STT)
- 📷 Real-time indoor object detection using YOLOv11n (TFLite)
- 🔊 Audio navigation feedback using Google Text-to-Speech (gTTS)
- ⚡ Runs completely offline on Raspberry Pi 5
- 🎒 Compact, wearable, and power-efficient design
- 💡 Directional logic: tells the user if the object is “in front,” “to your right,” etc.

---

## 🧠 Technologies Used

| Component           | Technology                     |
|--------------------|----------------------------------|
| Hardware           | Raspberry Pi 5, USB Webcam, AirPods |
| Object Detection   | YOLOv11n (custom & pre-trained) |
| Voice Recognition  | SpeechRecognition (Google STT) |
| Audio Feedback     | gTTS (Text-to-Speech)           |
| Programming        | Python 3                        |
| Vision Processing  | OpenCV                          |

---

## 🛠 Hardware Architecture

- **Raspberry Pi 5 (8GB)** – main processor
- **USB Webcam** – captures front-facing visual input
- **AirPods (Mic + Audio)** – for voice command and feedback
- **Cooling Fan + Protective Case** – ensures thermal stability
- **Portable Power Bank** – for mobile, wearable operation

---

## 📁 Project Structure

```bash
SightOfBlind/
├── models/              # YOLOv11n TFLite models (float16 quantized)
├── utils/               # Direction logic, preprocessing, helpers
├── main.py              # Main script for system execution
├── requirements.txt     # Python dependencies
├── media/               # Images, architecture diagrams, or demos
└── README.md            # Project overview
