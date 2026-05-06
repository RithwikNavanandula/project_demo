# Multi-Model OCR Web Application - Setup & Usage Guide

A full-stack Flask web application for text recognition from images using pre-trained CRNN-CTC models.

## 📋 Features

- **Image Upload**: Drag-and-drop or click-to-upload interface
- **Real-time Recognition**: Text extraction from images
- **Model Selection**: Choose `hindi` or `greek` at upload time
- **Two-Page Design**:
  - Page 1: Image upload interface
  - Page 2: Display uploaded image + recognized text
- **Copy to Clipboard**: Easy text copying
- **Mobile Responsive**: Works on desktop and mobile devices
- **Beautiful UI**: Modern gradient design with smooth animations

---

## 🚀 Quick Start

### 1. **Clone/Navigate to Project**
```bash
cd project_demo
```

### 2. **Create Virtual Environment (Linux/macOS)**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4. **Run the Application**
```bash
python app.py
```

The app will be available at: **http://localhost:5000**

---

## ☁️ Deploy on Render

This project is configured for Render using `render.yaml`.

### Option A: Blueprint Deploy (Recommended)
1. Push this repository to GitHub.
2. In Render dashboard, click **New +** → **Blueprint**.
3. Select your repo and deploy.
4. Render reads `render.yaml` and creates the web service automatically.

### Option B: Manual Web Service
Use these settings if creating a service manually:

- **Environment**: Python
- **Build Command**:
   ```bash
   pip install --upgrade pip && pip install -r requirements.txt
   ```
- **Start Command**:
   ```bash
   gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 300 app:app
   ```
- **Health Check Path**: `/api/models`

### Required Environment Variables
- `FLASK_ENV=production`
- `TF_USE_LEGACY_KERAS=1`
- `UPLOAD_FOLDER=/tmp/uploads`
- `PYTHON_VERSION=3.12.3`

Notes:
- Models are downloaded from Hugging Face at runtime if not present.
- First cold start can be slow due to model initialization.

### 5. **Windows Setup (PowerShell / CMD)**

#### Prerequisites
- Install **Python 3.10+** from https://www.python.org/downloads/windows/
- During install, enable **"Add Python to PATH"**

#### A) PowerShell
```powershell
cd project_demo
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

If activation is blocked, run this once in PowerShell (as current user):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### B) Command Prompt (CMD)
```cmd
cd project_demo
py -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open: **http://localhost:5000**

---

## 📁 Project Structure

```
project_demo/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── templates/
│   ├── upload.html          # Image upload page
│   └── predict.html         # Results display page
├── static/
│   ├── style.css            # Styling (responsive design)
│   ├── upload.js            # Upload page functionality
│   └── predict.js           # Results page functionality
├── uploads/                 # Directory for uploaded images (auto-created)
├── model_hindi/             # Hindi model files (auto-downloaded)
└── model_greek/             # Greek model files (auto-downloaded)
```

---

## 🎯 How It Works

### Frontend Flow
1. User opens **http://localhost:5000**
2. Uploads image via drag-drop or file picker
3. Flask processes image using selected OCR model
4. Results page displays image + recognized text
5. User can copy text or upload another image

### Backend Flow
1. **Model Download**: Automatically downloads from HuggingFace Hub
   - Hindi model: `rajesh-1902/hindi-ocr-crnn-ctc-v2`
   - Greek model: `rajesh-1902/greek-htr-crnn-ctc`
   - Files: Weights, preprocessor, model architecture
   
2. **Image Processing**:
   - Convert to grayscale
   - Preprocess using trained preprocessor
   - Pass through CRNN-CTC model
   
3. **Text Recognition**:
   - Model outputs text for the selected script
   - Results sent to frontend

---

## 📌 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Upload page |
| `/predict` | GET | Results page |
| `/api/upload` | POST | Handle image upload & prediction |
| `/api/status?model=hindi|greek` | GET | Check selected model status |
| `/api/models` | GET | List available models |

### Upload Endpoint Response
```json
{
  "success": true,
  "filename": "image.jpg",
  "filepath": "/static/../uploads/image.jpg",
   "predicted_text": "नमस्ते दुनिया",
   "model": "hindi"
}
```

---

## ⚙️ Configuration

### In `app.py`:
- **UPLOAD_FOLDER**: Where images are saved (default: `uploads/`)
- **MAX_CONTENT_LENGTH**: Max file size (default: 16MB)
- **MODEL_CONFIGS**: Per-model config for `hindi` and `greek`
- **ALLOWED_EXTENSIONS**: Image formats (png, jpg, jpeg, gif, bmp)

---

## 🔧 Model Details

- **Architecture**: CRNN-CTC (Convolutional Recurrent Neural Network + Connectionist Temporal Classification)
- **Input**: Grayscale images
- **Output**: Text from selected model (`hindi` or `greek`)
- **Hindi Model**: Devanagari OCR (`rajesh-1902/hindi-ocr-crnn-ctc-v2`)
- **Greek Model**: Greek HTR (`rajesh-1902/greek-htr-crnn-ctc`)
- **Download Size**: ~100MB+ per model (Greek weights are larger)
- **First Run**: Takes a few minutes per model (download + initialization)

---

## 🐛 Troubleshooting

### Problem: "Model not loading"
**Solution**: 
- Check internet connection (requires HuggingFace Hub access)
- Delete the affected model folder (`model_hindi/` or `model_greek/`) and restart app
- Check disk space (~200MB required)

### Problem: "Port 5000 already in use"
**Solution**:
```bash
# Change port in app.py (last line)
# app.run(debug=True, host='0.0.0.0', port=5001)
```

### Problem: "Import errors (cv2, tensorflow)"
**Solution**:
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Problem (Windows): "'python' is not recognized"
**Solution**:
- Use `py` instead of `python` in commands, or reinstall Python with **Add to PATH** enabled.

### Problem (Windows): "Activate.ps1 is disabled"
**Solution**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Problem: "Greek model first run is slow"
**Solution**:
- Expected on first run; large snapshot files are downloaded once.
- Keep the server running until initialization completes.

### Problem: "CUDA/GPU errors"
**Solution**: Set environment variable
```bash
CUDA_VISIBLE_DEVICES="" python app.py  # Force CPU only
```

---

## 💡 Tips for Best Results

✅ **Good Input**:
- Clear, well-lit images
- Straight, horizontal text
- Black text on white background
- Minimum resolution: 32px height
- Single line or paragraph text

❌ **Avoid**:
- Blurry or low-contrast images
- Rotated/tilted text
- Colored background
- Choosing wrong model for script (use `hindi` for Devanagari, `greek` for Greek)
- Very small text

---

## 🔐 Security Notes

- Files stored in `uploads/` folder
- Max file size: 16MB (configurable)
- Filenames sanitized using `secure_filename`
- Old files not auto-deleted (implement cleanup if needed)

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Flask | 3.0.0 | Web framework |
| TensorFlow | 2.19.0 | Model runtime |
| OpenCV | 4.8.0.74 | Image processing |
| NumPy | 1.26.4 | Array operations |
| HuggingFace Hub | 0.23.0 | Model download |

---

## 🔄 Version History

### v1.0 (Initial Release)
- Two-page upload and results interface
- Automatic model download
- Drag-drop file upload
- Copy to clipboard
- Mobile responsive design

---

## 📄 License

This project uses:
- **Hindi Model**: `rajesh-1902/hindi-ocr-crnn-ctc-v2` from HuggingFace Hub
- **Greek Model**: `rajesh-1902/greek-htr-crnn-ctc` from HuggingFace Hub
- **Framework**: Flask (BSD 3-Clause)

---

## 🙋 FAQ

**Q: Can I use this for production?**
A: Yes, but add authentication, rate limiting, and file cleanup. Use production WSGI server (Gunicorn, uWSGI).

**Q: How long does inference take?**
A: 1-2 seconds per image (after model warmup).

**Q: Can I train my own model?**
A: Yes, refer to the original HuggingFace repo for training code.

**Q: Which model names should I use?**
A: Use `hindi` for Hindi OCR and `greek` for Greek HTR.

**Q: Does it support other languages?**
A: Currently supports `hindi` and `greek`.

---

## 📞 Support

For Hindi model: https://huggingface.co/rajesh-1902/hindi-ocr-crnn-ctc-v2

For Greek model: https://huggingface.co/rajesh-1902/greek-htr-crnn-ctc

For Flask issues: https://flask.palletsprojects.com/

---

Happy OCR-ing! 🎉
