# 👁️ Diabetic Eye Monitor

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)

A comprehensive AI-driven system designed to monitor and detect early signs of diabetic eye complications. It achieves this by combining real-time blink analysis from webcam feeds with advanced retina image inference, calculating a dynamic risk score to provide timely recommendations.

---

## ✨ Features

* **Real-time Blink Analysis**: Utilizes webcams and Mediapipe face landmarks to track eye aspect ratio (EAR), blink duration, and calculate fatigue scores.
* **Retina Image Inference**: Analyzes static retina images for vessel clarity and focus scores using Tensorflow-based models.
* **Combined Risk Assessment**: A unified inference engine that fuses blink metrics and retina analysis to generate a diabetic eye risk level (LOW, MODERATE, HIGH).
* **Live Demo**: Ready-to-use OpenCV-based script (`live_demo.py`) that demonstrates the real-time capabilities of the system.
* **FastAPI Backend**: A robust REST API serving the inference models for easy integration with other systems.
* **Streamlit Dashboard**: An interactive, user-friendly dashboard for visualizing the metrics and risk assessments.

---

## 🏗️ Project Structure

```bash
.
├── api/                  # FastAPI backend containing routes and schemas
├── dashboard/            # Streamlit interactive dashboard
├── models/               # Pre-trained models for blink, retina, and fusion analysis
├── src/                  # Core source code
│   ├── blink_analysis/   # Modules for EAR calculation, blink metrics, and fatigue
│   ├── camera/           # Webcam streaming utilities
│   ├── explainability/   # Model interpretability tools
│   ├── inference/        # Centralized inference logic combining all metrics
│   ├── preprocessing/    # Data preparation utilities
│   ├── retina_analysis/  # Modules for focus scoring and vessel blur detection
│   ├── training/         # Model training scripts
│   └── utils/            # Shared configuration and helpers
├── tests/                # Unit testing suite (pytest)
├── live_demo.py          # Quickstart script for live OpenCV visualization
└── requirements.txt      # Project dependencies
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* A working webcam (for live blink analysis)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/diabetic-eye-monitor.git
   cd diabetic-eye-monitor
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### 1. Live OpenCV Demo
Experience the real-time blink analysis and risk scoring directly through your webcam:
```bash
python live_demo.py
```
*(Press `q` to quit the video stream)*

### 2. FastAPI Server
Start the backend API to serve the inference models:
```bash
cd api
uvicorn app:app --reload
```
The API documentation will be available at `http://localhost:8000/docs`.

### 3. Streamlit Dashboard
Launch the interactive dashboard for a complete view of the monitoring system:
```bash
cd dashboard
streamlit run streamlit_app.py
```

---

## 🧠 Under the Hood

The **CombinedInference** engine acts as the brain of the project. It evaluates two main components:
- **Fatigue Score**: Derived from real-time live webcam analysis monitoring the user's blink patterns.
- **Retina Metrics**: Calculated from static retina images, focusing on average focus score and vessel clarity.

A weighted formula is then applied to these metrics to calculate the final `risk_score`, which correlates to a `risk_level` with actionable recommendations.

---

## 🧪 Testing

The repository uses `pytest` for unit testing. To run the test suite:
```bash
pytest tests/
```

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*Disclaimer: This tool is intended for research and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.*
