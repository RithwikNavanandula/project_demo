"""
Flask web application for Hindi OCR with image upload and prediction
"""
import os
import sys
import shutil
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from huggingface_hub import hf_hub_download
from config import app_config

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
UPLOAD_FOLDER = app_config.UPLOAD_FOLDER
ALLOWED_EXTENSIONS = app_config.ALLOWED_EXTENSIONS
# Use absolute paths — mirrors the notebook's /content/model_hindi pattern
CODE_DIR = os.path.abspath(os.getcwd())
MODEL_DIR = os.path.join(CODE_DIR, 'model_hindi')
MODEL_REPO_ID = app_config.MODEL_REPO_ID

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__)
app.config.from_object(app_config)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables for model
model = None
charList = None
model_initialized = False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ═══════════════════════════════════════════════════════════════
# MODEL SETUP
# ═══════════════════════════════════════════════════════════════
def download_model():
    """Download model from HuggingFace Hub"""
    print("\n📥 Downloading model weights (snapshot-5)...")
    
    REPO_ID = "rajesh-1902/hindi-crnn-ctc-sentence-model"
    
    model_files = [
        "model/charList.txt",
        "model/corpus.txt",
        "model/accuracy.txt",
        "model/snapshot-5.data-00000-of-00001",
        "model/snapshot-5.index",
        "model/snapshot-5.meta",
    ]
    
    for hf_path in model_files:
        try:
            local = hf_hub_download(repo_id=REPO_ID, filename=hf_path, repo_type="model")
            shutil.copy2(local, os.path.join(MODEL_DIR, os.path.basename(hf_path)))
            print(f"  ✅ {os.path.basename(hf_path)}")
        except Exception as e:
            print(f"  ❌ Error downloading {hf_path}: {e}")
            return False
    
    print("\n📥 Downloading code files...")
    code_files = [
        "code/Model_Hindi.py",
        "code/DataLoader_Hindi.py",
        "code/SamplePreprocessor_Hindi.py",
        "code/main_hindi.py",
        "code/build_charlist_hindi.py",
    ]
    
    for hf_path in code_files:
        try:
            local = hf_hub_download(repo_id=REPO_ID, filename=hf_path, repo_type="model")
            shutil.copy2(local, os.path.join(CODE_DIR, os.path.basename(hf_path)))
            print(f"  ✅ {os.path.basename(hf_path)}")
        except Exception as e:
            print(f"  ❌ Error downloading {hf_path}: {e}")
            return False
    
    return True


def setup_checkpoint():
    """Write checkpoint file with absolute path — mirrors notebook exactly"""
    SNAPSHOT = os.path.join(MODEL_DIR, "snapshot-5")  # absolute path
    with open(os.path.join(MODEL_DIR, "checkpoint"), "w") as f:
        f.write(f'model_checkpoint_path: "{SNAPSHOT}"\n')
        f.write(f'all_model_checkpoint_paths: "{SNAPSHOT}"\n')
    print(f"✅ checkpoint → {SNAPSHOT}")


def fix_model_bugs():
    """Fix known bugs in model files"""
    model_py = os.path.join(CODE_DIR, "Model_Hindi.py")
    
    if os.path.exists(model_py):
        with open(model_py, "r") as f:
            src = f.read()
        src = src.replace("tv.initialized_value()", "tv")
        src = src.replace("../model_hindi/", "model_hindi/")
        with open(model_py, "w") as f:
            f.write(src)
    
    for fname in ["main_hindi.py"]:
        fp = os.path.join(CODE_DIR, fname)
        if os.path.exists(fp):
            with open(fp, "r") as f:
                c = f.read()
            c = c.replace("../model_hindi/", "model_hindi/")
            with open(fp, "w") as f:
                f.write(c)
    
    print("✅ Bug fixes applied")


def load_model_and_charlist():
    """Load the Hindi OCR model — mirrors notebook exactly"""
    global model, charList, model_initialized
    
    if model is not None and charList is not None:
        return True
    
    try:
        print("\n🔧 Setting up TensorFlow...")
        # Mirrors notebook FINAL FIX CELL exactly
        tf.keras.backend.clear_session()
        tf.compat.v1.disable_eager_execution()
        
        # Stay in CODE_DIR — exactly like notebook's os.chdir(CODE_DIR)
        os.chdir(CODE_DIR)
        
        # Force fresh imports
        for mod in ["Model_Hindi", "SamplePreprocessor_Hindi", "DataLoader_Hindi"]:
            sys.modules.pop(mod, None)
        
        sys.path.insert(0, CODE_DIR)
        from Model_Hindi import Model, DecoderType
        from SamplePreprocessor_Hindi import preprocess
        
        # Read charList using absolute path (MODEL_DIR is absolute)
        with open(os.path.join(MODEL_DIR, "charList.txt"), "r", encoding="utf-8") as f:
            charList = f.read()
        print(f"✅ Character set loaded: {len(charList)} chars")
        
        # Instantiate model — checkpoint has absolute path so TF finds it correctly
        model = Model(charList, DecoderType.BestPath, mustRestore=True)
        print("✅ Model restored successfully!")
        model_initialized = True
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        model_initialized = False
        return False


def recognize(image_path: str) -> str:
    """Run Hindi OCR on image"""
    global model, charList
    
    try:
        sys.path.insert(0, CODE_DIR)
        from SamplePreprocessor_Hindi import preprocess
        from Model_Hindi import Model
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")
        
        img_proc = preprocess(img, Model.imgSize)
        batch = np.expand_dims(img_proc, axis=0)
        
        class _Batch:
            def __init__(self, imgs):
                self.imgs = imgs
        
        result = model.inferBatch(_Batch(batch))[0]
        return result
    except Exception as e:
        print(f"❌ Error during recognition: {e}")
        raise


# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════
@app.route('/')
def index():
    """Upload page"""
    return render_template('upload.html')


@app.route('/predict')
def predict_page():
    """Prediction results page"""
    return render_template('predict.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle image upload and return prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"\n📸 Processing: {filename}")
        
        # Check model status
        if model is None or not model_initialized:
            return jsonify({'error': 'Model not loaded. Please wait for initialization or restart the application.'}), 503
        
        # Run prediction
        try:
            predicted_text = recognize(filepath)
            print(f"✅ Prediction: {predicted_text}")
            
            return jsonify({
                'success': True,
                'filename': filename,
                'filepath': url_for('static', filename=f'../uploads/{filename}'),
                'predicted_text': predicted_text
            }), 200
        except Exception as e:
            print(f"❌ Recognition error: {e}")
            return jsonify({'error': f'Recognition failed: {str(e)}'}), 500
    
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/api/status')
def status():
    """Check model status"""
    return jsonify({
        'model_loaded': model is not None,
        'charlist_loaded': charList is not None
    })


# ═══════════════════════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════════════════════

def initialize_model():
    """Initialize model on app startup"""
    global model, charList, model_initialized
    
    if model_initialized:
        return
    
    print("\n🚀 Initializing model...")
    
    # Check if ALL model files exist (not just charList.txt)
    model_files_exist = (
        os.path.exists(os.path.join(MODEL_DIR, "charList.txt")) and
        os.path.exists(os.path.join(MODEL_DIR, "snapshot-5.data-00000-of-00001")) and
        os.path.exists(os.path.join(MODEL_DIR, "snapshot-5.index")) and
        os.path.exists(os.path.join(MODEL_DIR, "snapshot-5.meta"))
    )
    
    # Download model if not ALL files are present
    if not model_files_exist:
        print("📥 Model files not found locally, downloading from HuggingFace Hub...")
        print("   This may take 2-5 minutes on first run (~100MB download)")
        if not download_model():
            print("❌ Failed to download model")
            return False
    else:
        print("✅ Model files found locally")
    
    setup_checkpoint()
    fix_model_bugs()
    
    if not load_model_and_charlist():
        print("❌ Failed to load model")
        return False
    
    print("✅ Model initialized successfully!")
    return True


@app.before_request
def before_request():
    """Check model status on each request"""
    global model_initialized
    
    # Initialize if not done yet
    if not model_initialized:
        initialize_model()


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Page not found'}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    print(f"❌ Server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("🌐 Starting Hindi OCR Web Application...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📁 Model directory: {MODEL_DIR}")
    print(f"🔗 Visit: http://localhost:5000")
    print("")
    
    # Initialize model before starting server
    initialize_model()
    
    # Use debug mode from config
    app.run(debug=app_config.DEBUG, host='0.0.0.0', port=5000, use_reloader=False)
