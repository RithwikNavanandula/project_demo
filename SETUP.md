# Hindi OCR Web Application - Complete Setup Guide

## 📋 Overview

This is a complete local Hindi OCR (Optical Character Recognition) web application that converts images of Hindi text to digital text. Built with Flask, TensorFlow, and modern web technologies.

---

## 🎯 Quick Start (5 Minutes)

### Option 1: Using Shell Script (Linux/Mac)
```bash
cd /home/rishi/esc/COLLEGE/final-year-project/project_demo
chmod +x run.sh
./run.sh
```
Then open: **http://localhost:5000**

### Option 2: Manual Setup (Linux/Mac/Windows)
```bash
# 1. Navigate to project directory
cd /home/rishi/esc/COLLEGE/final-year-project/project_demo

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the application
python app.py
```

**Open your browser and go to:** http://localhost:5000

---

## 📦 What Gets Downloaded Automatically?

When you run the app for the first time, it automatically downloads:

1. **Hindi OCR Model** (~100MB)
   - From: HuggingFace Hub (rajesh-1902/hindi-ocr-crnn-ctc-v2)
   - Includes: Model weights, preprocessor, character list
   
2. **Required Python Packages** (specified in requirements.txt)
   - TensorFlow (deep learning framework)
   - OpenCV (image processing)
   - Flask (web framework)
   - And more...

**Total download:** ~500MB-1GB (depends on system)
**First run time:** 2-3 minutes (model download + initialization)

---

## 🏗️ Project Structure Explained

```
project_demo/
│
├── app.py                          # Main Flask application
│   ├── Model download logic
│   ├── Image processing
│   ├── API routes (/api/upload, /api/status)
│   └── Error handling
│
├── config.py                       # Configuration management
│   ├── DevelopmentConfig
│   ├── ProductionConfig
│   └── TestingConfig
│
├── templates/                      # HTML pages
│   ├── upload.html                # Page 1: Image upload interface
│   └── predict.html               # Page 2: Results display
│
├── static/                         # Frontend assets
│   ├── style.css                  # Responsive styling
│   ├── upload.js                  # Upload page logic
│   └── predict.js                 # Results page logic
│
├── uploads/                        # Uploaded images (auto-created)
│
├── model_hindi/                    # Model files (auto-created & downloaded)
│
├── requirements.txt                # Python dependencies
├── run.sh                          # Startup script
├── Dockerfile                      # Docker image configuration
├── docker-compose.yml              # Multi-container setup
├── README.md                       # Main documentation
└── SETUP.md                        # This file
```

---

## 🎨 How the Application Works

### **Page 1: Upload Page** (`/`)
1. User visits http://localhost:5000
2. Options to upload image:
   - Click "Choose Image" button
   - Drag and drop image into drop zone
3. Click "Upload & Recognize" button
4. Shows loading spinner while processing
5. Automatically redirects to results page

### **Page 2: Results Page** (`/predict`)
1. Displays uploaded image
2. Shows recognized Hindi text
3. Options:
   - **Copy Text**: Copy recognized text to clipboard
   - **Upload Another**: Go back to upload page

---

## 🔌 API Endpoints

### 1. Upload and Predict
```
POST /api/upload
Content-Type: multipart/form-data

Body: file (image file)

Response:
{
  "success": true,
  "filename": "image.jpg",
  "filepath": "/uploads/image.jpg",
  "predicted_text": "नमस्ते दुनिया"
}
```

### 2. Check Model Status
```
GET /api/status

Response:
{
  "model_loaded": true,
  "charlist_loaded": true
}
```

### 3. Get Upload Page
```
GET /
Response: HTML page with upload interface
```

### 4. Get Results Page
```
GET /predict
Response: HTML page with results (requires session data)
```

---

## ⚙️ Configuration

All settings in `config.py`:

```python
UPLOAD_FOLDER = 'uploads'              # Where images are saved
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # Max file size: 16MB
ALLOWED_EXTENSIONS = {
    'png', 'jpg', 'jpeg', 'gif', 'bmp'
}
MODEL_DIR = 'model_hindi'               # Model storage directory
MODEL_REPO_ID = 'rajesh-1902/hindi-ocr-crnn-ctc-v2'
```

---

## 🐛 Common Issues & Solutions

### Issue 1: "ModuleNotFoundError: No module named 'flask'"
**Solution:** Run `pip install -r requirements.txt` again
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Issue 2: "Address already in use" on port 5000
**Solution:** Change port in app.py or kill the process
```bash
# On Linux/Mac - kill process on port 5000
lsof -ti:5000 | xargs kill -9

# On Windows - find process and kill it
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Issue 3: Model download fails
**Solution:** Check internet connection and try again
```bash
# Delete partial download and restart
rm -rf model_hindi/
python app.py
```

### Issue 4: TensorFlow/GPU errors
**Solution:** Force CPU mode
```bash
CUDA_VISIBLE_DEVICES="" python app.py
```

### Issue 5: Image not processing
**Solution:** Ensure image is:
- Clear and legible
- In supported format (PNG, JPG, etc.)
- Not too large (< 16MB)
- Contains Hindi text

---

## 💾 Production Deployment

### Using Docker

1. **Build Docker image:**
   ```bash
   docker build -t hindi-ocr:latest .
   ```

2. **Run with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

3. **Access at:** http://localhost:5000

### Using Gunicorn (Production Server)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using uWSGI

```bash
pip install uwsgi
uwsgi --http :5000 --wsgi-file app.py --callable app
```

---

## 🚀 Advanced Usage

### Environment Variables

Create `.env` file:
```
FLASK_ENV=production
UPLOAD_FOLDER=uploads
MODEL_DIR=model_hindi
```

### Custom Model Path

```python
# In config.py
MODEL_DIR = os.getenv('MODEL_DIR', '/custom/path/model_hindi')
```

### Enable HTTPS

```python
# In app.py
app.run(ssl_context='adhoc')  # Requires pyopenssl
```

---

## 📊 Performance Tips

1. **Faster Recognition:**
   - Use smaller, clear images
   - Ensure good contrast
   - Single line of text (if possible)

2. **Better Memory Usage:**
   - Set `use_reloader=False` in production
   - Use process managers (supervisor, systemd)

3. **Scaling:**
   - Use load balancer (nginx)
   - Run multiple instances
   - Use task queue (Celery) for long processing

---

## 🔐 Security Considerations

1. **File Upload Safety:**
   - Files are saved with sanitized names
   - Max size limit: 16MB
   - Only specific file types allowed

2. **For Production:**
   - Use HTTPS/SSL
   - Implement authentication
   - Add rate limiting
   - Use Web Application Firewall (WAF)

3. **Database:**
   - Currently uses file system
   - For production, use database (PostgreSQL)

---

## 📝 Customization Guide

### Change Maximum File Size

In `config.py`:
```python
MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32MB
```

### Add New Allowed File Types

In `config.py`:
```python
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'}
```

### Modify UI Colors

In `static/style.css`:
```css
:root {
    --primary-color: #ff5733;  /* Change this to your color */
    --secondary-color: #c70039;
}
```

### Change Model

In `config.py`:
```python
MODEL_REPO_ID = 'your-username/your-model-name'
```

---

## 🧪 Testing

Run tests:
```bash
python -m pytest tests/
```

Test specific endpoint:
```bash
curl -X POST -F "file=@test_image.jpg" http://localhost:5000/api/upload
```

---

## 📞 Support Resources

- **TensorFlow Issues**: https://stackoverflow.com/questions/tagged/tensorflow
- **Flask Documentation**: https://flask.palletsprojects.com/
- **OpenCV Docs**: https://docs.opencv.org/
- **HuggingFace Model**: https://huggingface.co/rajesh-1902/hindi-ocr-crnn-ctc-v2

---

## 📄 License & Attribution

- **Model**: rajesh-1902/hindi-ocr-crnn-ctc-v2 (HuggingFace)
- **Framework**: Flask (BSD 3-Clause License)
- **Libraries**: TensorFlow, OpenCV, NumPy (Apache 2.0 / MIT)

---

## 🎓 Learning Resources

- **How OCR Works**: https://en.wikipedia.org/wiki/Optical_character_recognition
- **CRNN Architecture**: https://arxiv.org/abs/1507.05717
- **CTC Loss**: https://arxiv.org/abs/1311.4508

---

## 🤝 Contributing

Feel free to fork and contribute improvements!

---

**Happy OCR-ing! 🎉**

Last Updated: April 2026
