# 🛡️ Image Provenance Vision Transformer Deepfake Detection

An advanced deepfake detection and image provenance analysis system powered by Vision Transformers and graph-based encoding. This project combines modern AI research with a scalable web architecture to detect manipulated images and trace their origins.

---

## 🚀 Overview

This project integrates:
🧠 Vision Transformers (ViT) for deepfake detection

🔗 Graph-based encoding for image provenance tracking

⚡ FastAPI / Flask backend services for model inference

🌐 Next.js frontend (App Router) for a modern web interface

🗄️ SQLite database for demo and temporary storage

The system enables users to upload images, analyze authenticity, and visualize provenance relationships.

---

## 📁 Project Structure
``` bash

Image Provenance Vision Transformer Deepfake Detection/
├── api/                  # Python-based FastAPI backend services for vision transformers and graph encoding
├── app/                  # Next.js App Directory
├── backend/              # Python-based backend services (Flask/FastAPI)
├── components/           # Reusable UI Components
├── hooks/                # React custom hooks
├── lib/                  # Utility libraries
├── public/               # Static assets
├── scripts/              # Automation or setup scripts
├── styles/               # TailwindCSS and global styles
├── socialguard.db        # SQLite database (temporary/demo)
├── *.config.mjs          # Configuration files (Next.js, PostCSS)
├── *.json                # Project metadata and dependencies

```
---

## 🧠 Core Features

### 🔍 Deepfake Detection

Vision Transformer-based image classification
Real vs manipulated prediction
Confidence score output
Model inference via API

### 🕸️ Image Provenance Analysis

Graph-based encoding for relationship modeling
Visual representation of image transformations
Tracking possible source images

### 🌐 Web Application

Built with Next.js (App Router)
Modern UI with TailwindCSS
REST API integration with backend AI services
Modular and scalable architecture

---

## Architecture Overview

```bash
Frontend (Next.js)
        ↓
API Layer (FastAPI / Flask)
        ↓
Vision Transformer Model
        ↓
Graph Encoding Module
        ↓
SQLite Database (Demo)
```

---

## Installation & Setup

### 1️⃣ Clone the Repository
``` bash
git clone https://github.com/ksh-20/Image_Provenance
cd Image_Provenance
```

### 2️⃣ Backend Setup (Python)
Create a virtual environment:
``` bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate      # Windows
```

Install dependencies:
``` bash
pip install -r requirements.txt
```

Run the backend:
``` bash
uvicorn api.main:app --reload
```

Or if using Flask:
``` bash
python app.py
```

### 3️⃣ Frontend Setup (Next.js)
Install dependencies:
``` bash
npm install
```

Run development server:
``` bash
npm run dev
```

App will be available at:
``` bash
http://localhost:3000
```

---

## 🗄️ Database

The project includes a temporary SQLite database:
socialguard.db

Used for:
Storing uploaded image metadata
Storing detection results
Tracking provenance graph references

---

## 📦 Tech Stack

### Frontend

Next.js (App Router)
React
TailwindCSS

### Backend

FastAPI / Flask
PyTorch (Vision Transformer models)
Graph encoding libraries

### Database

SQLite (demo)

---

## 🧪 Example API Endpoint

``` bash
POST /analyze-image
Content-Type: multipart/form-data
```

Response:
``` JSON
{
  "prediction": "deepfake",
  "confidence": 0.97,
  "provenance_graph": [...]
}
```

---

## 🔐 Future Improvements

🔄 Model fine-tuning with larger datasets

📊 Admin dashboard with analytics

☁️ Cloud deployment (AWS/GCP/Azure)

🧬 Multi-modal deepfake detection (video + audio)

🗃️ Replace SQLite with production-grade database

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

## 👨‍💻 Author

Developed for research and educational purposes in AI-powered deepfake detection and image provenance tracking.

---