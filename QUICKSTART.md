# 🚀 Hindi OCR Web App - Quick Start (2 Minutes)

## Step 1: Navigate to Project
```bash
cd /home/rishi/esc/COLLEGE/final-year-project/project_demo
```

## Step 2: Run the App

### Option A: Automatic (Recommended for Linux/Mac)
```bash
chmod +x run.sh
./run.sh
```

### Option B: Manual
```bash
# Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run app
python app.py
```

## Step 3: Open Browser
Visit: **http://localhost:5000**

---

## ✨ What You'll See

### Page 1: Upload Page
- Drag & drop image or click to browse
- Supported formats: PNG, JPG, JPEG, GIF, BMP
- Max size: 16MB

### Page 2: Results Page
- Your uploaded image displayed
- Recognized Hindi text shown
- Option to copy text
- Option to upload another image

---

## ⏱️ First Time Setup

First run takes **2-3 minutes** because:
1. Model download (~100MB) - happens once
2. Model initialization
3. Loading dependencies

Subsequent runs are **instant**! ⚡

---

## 🛑 Stop the Server

Press: **Ctrl + C** in your terminal

---

## 📁 File Structure Created

```
✅ app.py                 - Main application
✅ config.py             - Configuration
✅ requirements.txt      - Dependencies
✅ run.sh                - Startup script
✅ templates/            - HTML pages
✅ static/               - CSS & JavaScript
✅ uploads/              - Uploaded images (auto-created)
✅ model_hindi/          - Model files (auto-downloaded)
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 5000 in use | Change port in app.py or kill process |
| Model won't download | Check internet, delete model_hindi/, restart |
| Import errors | Run `pip install -r requirements.txt` again |
| Image not recognized | Ensure image has clear Hindi text |

---

## 📚 Full Documentation

For detailed setup, configuration, and troubleshooting:
- **SETUP.md** - Complete setup guide
- **README.md** - Full documentation
- **config.py** - Configuration options

---

**That's it! 🎉 Your Hindi OCR app is ready to use!**
