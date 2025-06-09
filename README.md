
# 🧬 Vitamin Deficiency Detection Web App

This is a full-stack web application for detecting vitamin deficiencies using skin image classification. It uses a PyTorch model on the Flask backend and a React frontend for user interaction.

---

## 🛠 Tech Stack

| Layer     | Tech                         |
|-----------|------------------------------|
| Backend   | Flask + PyTorch              |
| Frontend  | React.js                     |
| ML Model  | CNN (.pth file)              |
| Dev Tools | VS Code, Node.js, Python 3.8 |

---

## 📁 Folder Structure

```
vitamin-deficiency-webapp/
├── backend/
│   ├── app.py                  # Flask backend
│   ├── best_vitamin_model2.pth # Trained PyTorch model
├── frontend/                   # React frontend
│   ├── public/
│   └── src/
```

---

## 🚀 How to Run the Project

### 🔹 1. Run the Backend (Flask + PyTorch)

```bash
cd backend
python -m venv venv
venv/Scripts/activate         # On Windows
# or
source venv/bin/activate      # On Mac/Linux

pip install flask torch torchvision
python app.py
```

> 🔌 Backend runs at: `http://127.0.0.1:5000`

---

### 🔹 2. Run the Frontend (React)

```bash
cd ../frontend
npm install
npm start
```

> 🌐 Frontend runs at: `http://localhost:3000`

---

## 🧪 How It Works

- User uploads an image from the frontend
- Flask backend receives it via `/predict`
- The trained PyTorch model makes a prediction
- Result (e.g. Vitamin D Deficiency) is sent back and displayed

---

## ⚠️ Notes

- PyTorch warning about `weights_only=False` is safe to ignore if you trust the model.
- Make sure CORS and API paths are correctly configured between frontend and backend.


