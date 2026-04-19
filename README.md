# Hindi OCR Web Application - Setup & Usage Guide

A full-stack Flask web application for Hindi text recognition from images using a pre-trained CRNN-CTC model.

## 📋 Features

- **Image Upload**: Drag-and-drop or click-to-upload interface
- **Real-time Recognition**: Hindi text extraction from images
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
cd /home/rishi/esc/COLLEGE/final-year-project/project_demo
```

### 2. **Create Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
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
└── model_hindi/             # Model files (auto-downloaded)
```

---

## 🎯 How It Works

### Frontend Flow
1. User opens **http://localhost:5000**
2. Uploads image via drag-drop or file picker
3. Flask processes image and runs OCR model
4. Results page displays image + recognized Hindi text
5. User can copy text or upload another image

### Backend Flow
1. **Model Download**: Automatically downloads from HuggingFace Hub
   - Model: `rajesh-1902/hindi-crnn-ctc-sentence-model`
   - Files: Weights, preprocessor, model architecture
   
2. **Image Processing**:
   - Convert to grayscale
   - Preprocess using trained preprocessor
   - Pass through CRNN-CTC model
   
3. **Text Recognition**:
   - Model outputs Hindi text
   - Results sent to frontend

---

## 📌 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Upload page |
| `/predict` | GET | Results page |
| `/api/upload` | POST | Handle image upload & prediction |
| `/api/status` | GET | Check model status |

### Upload Endpoint Response
```json
{
  "success": true,
  "filename": "image.jpg",
  "filepath": "/static/../uploads/image.jpg",
  "predicted_text": "नमस्ते दुनिया"
}
```

---

## ⚙️ Configuration

### In `app.py`:
- **UPLOAD_FOLDER**: Where images are saved (default: `uploads/`)
- **MAX_CONTENT_LENGTH**: Max file size (default: 16MB)
- **MODEL_DIR**: Where model is stored (default: `model_hindi/`)
- **ALLOWED_EXTENSIONS**: Image formats (png, jpg, jpeg, gif, bmp)

---

## 🔧 Model Details

- **Architecture**: CRNN-CTC (Convolutional Recurrent Neural Network + Connectionist Temporal Classification)
- **Input**: Grayscale images
- **Output**: Hindi text (Devanagari script)
- **Characters**: 100+ Hindi characters supported
- **Download Size**: ~100MB
- **First Run**: Takes 2-3 minutes (model download + initialization)

---

## 🐛 Troubleshooting

### Problem: "Model not loading"
**Solution**: 
- Check internet connection (requires HuggingFace Hub access)
- Delete `model_hindi/` folder and restart app
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
- Mixed scripts (Hindi + English)
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
| Flask | 2.3.3 | Web framework |
| TensorFlow | 2.13.0 | Model runtime |
| OpenCV | 4.8.0 | Image processing |
| NumPy | 1.24.3 | Array operations |
| HuggingFace Hub | 0.16.4 | Model download |

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
- **Model**: `rajesh-1902/hindi-crnn-ctc-sentence-model` from HuggingFace Hub
- **Framework**: Flask (BSD 3-Clause)

---

## 🙋 FAQ

**Q: Can I use this for production?**
A: Yes, but add authentication, rate limiting, and file cleanup. Use production WSGI server (Gunicorn, uWSGI).

**Q: How long does inference take?**
A: 1-2 seconds per image (after model warmup).

**Q: Can I train my own model?**
A: Yes, refer to the original HuggingFace repo for training code.

**Q: Does it support other languages?**
A: No, only Hindi. Other models available on HuggingFace Hub.

---

## 📞 Support

For issues with the model, visit: https://huggingface.co/rajesh-1902/hindi-crnn-ctc-sentence-model

For Flask issues: https://flask.palletsprojects.com/

---

Happy OCR-ing! 🎉
# project_demo
