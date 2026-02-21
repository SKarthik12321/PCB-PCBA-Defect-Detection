# 🎛️ PCB & PCBA Defect Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![React 18](https://img.shields.io/badge/React-18.0+-61dafb.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)

A state-of-the-art framework for automated Printed Circuit Board (PCB) and PCBA defect detection.

![Premium Dark UI - Main Interface](./pcb_final_verify_1771698843986.png)

## 🚀 Overview

This repository contains a complete, robust, and real-time inspection system designed to identify manufacturing defects on PCBs. The system leverages an advanced deep learning architecture to achieve high-precision defect classification and localization. 

The frontend provides a premium, industrial-grade web interface featuring live webcam inference and automatic drag-and-drop image analysis, built with React and Vite. The backend is powered by FastAPI and PyTorch.

## 🧠 Core Architecture & Algorithms

Our detection framework is built upon a novel, research-grade architecture optimized for the unique challenges of PCB inspection (e.g., microscopic defect sizes, extremely complex background circuit structures).

* **Geometry-Aware Backbone:** Extracts rich, high-resolution spatial features while maintaining strict sensitivity to the intricate geometric patterns of PCB traces, vias, and pads.
* **DACSR (Domain Adaptive Cross-Scale Representation):** Facilitates heavy multi-scale feature fusion, enabling the network to accurately detect defects varying drastically in size—from microscopic mouse bites to large open circuits or shorts.
* **CARFT (Context-Aware Receptive Field Tuning):** Dynamically adjusts the receptive field based on the local component context, significantly reducing false positives in densely populated areas of the PCBA.
* **YOLO11 Detection Head:** A highly optimized, state-of-the-art detection head used for real-time bounding box regression and classification, achieving ultra-fast inference speeds perfect for the live webcam streaming feature.

## 🔍 Supported Defect Classes

The included models have been meticulously trained to recognize 6 critical PCB defect classes:
1. 🔴 **Missing Hole**
2. 🟠 **Mouse Bite**
3. 🟡 **Open Circuit**
4. 🟣 **Short**
5. 🔵 **Spur**
6. 🟢 **Spurious Copper**

## 🛠️ Technology Stack

* **Frontend:** React 18, Vite, Lucide React (SVG icons), custom Glassmorphism CSS UI, dynamic SVG trace animations.
* **Backend:** FastAPI, Uvicorn, Python 3.9+
* **Deep Learning AI:** PyTorch, Ultralytics (YOLO11)
* **Computer Vision:** OpenCV (`cv2`), NumPy

## 📦 Setup & Installation

Follow these steps to run the inspection system locally.

### 1. Backend Setup

First, initialize the Python backend to serve the model and API routes.

```bash
# Navigate to the project root
cd PCB-PCBA-Defect-Detection

# Install Python requirements
pip install -r requirements.txt

# Start the FastAPI backend server
python -m uvicorn webapp.backend.app:app --port 8000 --reload
```

### 2. Frontend Setup

In a new terminal, build and run the React frontend.

```bash
# Navigate to the frontend directory
cd webapp/react-frontend

# Install Node dependencies
npm install

# Start the Vite development server (proxies to backend)
npm run dev
```

Visit `http://localhost:5173/` in your browser to access the inspection system!

## 🤖 Pre-Trained Models

The repository includes the fully trained `pcb_model.pt` weights required for immediate deployment. Place the model file inside the `models/` directory if it is not already present. The FastAPI backend will automatically discover and load the model on startup.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue if you encounter bugs or want to request a feature.