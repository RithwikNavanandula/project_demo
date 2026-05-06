"""
Flask web application for OCR with multi-model support (Hindi + Greek)
"""
import inspect
import importlib.util
import os
import re
import shutil
import sys
from dataclasses import dataclass

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from huggingface_hub import hf_hub_download, list_repo_files
from werkzeug.utils import secure_filename

from config import app_config

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
UPLOAD_FOLDER = app_config.UPLOAD_FOLDER
ALLOWED_EXTENSIONS = app_config.ALLOWED_EXTENSIONS
CODE_DIR = os.path.abspath(os.getcwd())
DEFAULT_MODEL_KEY = "hindi"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@dataclass
class ModelConfig:
    key: str
    display_name: str
    repo_id: str
    model_dir: str
    snapshot_prefix_remote: str
    code_files: list[tuple[str, str]]
    required_artifacts: list[str]
    optional_artifacts: list[str]
    model_file_local: str
    preprocessor_file_local: str


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "hindi": ModelConfig(
        key="hindi",
        display_name="Hindi OCR",
        repo_id="rajesh-1902/hindi-ocr-crnn-ctc-v2",
        model_dir=os.path.join(CODE_DIR, "model_hindi"),
        snapshot_prefix_remote="model_hindi/snapshot-",
        code_files=[
            ("Model_Hindi_v2.py", "Model_Hindi_v2.py"),
            ("DataLoader_Hindi_v2.py", "DataLoader_Hindi_v2.py"),
            ("SamplePreprocessor_Hindi_v2.py", "SamplePreprocessor_Hindi_v2.py"),
            ("main_hindi_v2.py", "main_hindi_v2.py"),
            ("build_charlist_hindi.py", "build_charlist_hindi.py"),
        ],
        required_artifacts=["charList.txt", "checkpoint", "accuracy.txt"],
        optional_artifacts=["metrics.json", "corpus.txt"],
        model_file_local="Model_Hindi_v2.py",
        preprocessor_file_local="SamplePreprocessor_Hindi_v2.py",
    ),
    "greek": ModelConfig(
        key="greek",
        display_name="Greek HTR",
        repo_id="rajesh-1902/greek-htr-crnn-ctc",
        model_dir=os.path.join(CODE_DIR, "model_greek"),
        snapshot_prefix_remote="snapshot-",
        code_files=[
            ("src/Model.py", "Model_Greek.py"),
            ("src/DataLoader.py", "DataLoader_Greek.py"),
            ("src/SamplePreprocessor.py", "SamplePreprocessor_Greek.py"),
            ("src/main.py", "main_Greek.py"),
            ("src/build_charlist.py", "build_charlist_greek.py"),
        ],
        required_artifacts=["charList.txt", "checkpoint", "accuracy.txt"],
        optional_artifacts=["corpus.txt", "README.md"],
        model_file_local="Model_Greek.py",
        preprocessor_file_local="SamplePreprocessor_Greek.py",
    ),
    "greek-word": ModelConfig(
        key="greek-word",
        display_name="Greek HTR (Word)",
        repo_id="rithwikn/greek-historical-htr-model",
        model_dir=os.path.join(CODE_DIR, "model_greek_word"),
        snapshot_prefix_remote="snapshot-",
        code_files=[
            ("Model.py", "Model_GreekWord.py"),
            ("SamplePreprocessor.py", "SamplePreprocessor_GreekWord.py"),
        ],
        required_artifacts=["charList.txt", "checkpoint"],
        optional_artifacts=[],
        model_file_local="Model_GreekWord.py",
        preprocessor_file_local="SamplePreprocessor_GreekWord.py",
    ),
}

for cfg in MODEL_CONFIGS.values():
    os.makedirs(cfg.model_dir, exist_ok=True)

app = Flask(__name__)
app.config.from_object(app_config)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Runtime state per model
MODEL_STATES = {
    key: {
        "model": None,
        "charList": None,
        "initialized": False,
        "initializing": False,
        "model_class": None,
        "preprocess_fn": None,
    }
    for key in MODEL_CONFIGS
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_model_key(value: str | None) -> str:
    candidate = (value or DEFAULT_MODEL_KEY).strip().lower()
    return candidate if candidate in MODEL_CONFIGS else DEFAULT_MODEL_KEY


def get_latest_snapshot_prefix(model_dir: str):
    snapshot_ids = []
    for file_name in os.listdir(model_dir):
        match = re.match(r"snapshot-(\d+)\.index$", file_name)
        if match:
            snapshot_ids.append(int(match.group(1)))
    if not snapshot_ids:
        return None
    latest_id = max(snapshot_ids)
    return os.path.join(model_dir, f"snapshot-{latest_id}")


def setup_checkpoint(cfg: ModelConfig):
    snapshot = get_latest_snapshot_prefix(cfg.model_dir)
    if snapshot is None:
        raise FileNotFoundError(f"No snapshot-*.index files found in {cfg.model_dir}")

    checkpoint_file = os.path.join(cfg.model_dir, "checkpoint")
    with open(checkpoint_file, "w", encoding="utf-8") as file_obj:
        file_obj.write(f'model_checkpoint_path: "{snapshot}"\n')
        file_obj.write(f'all_model_checkpoint_paths: "{snapshot}"\n')
    print(f"✅ [{cfg.key}] checkpoint → {snapshot}")


def download_model(cfg: ModelConfig):
    print(f"\n📥 [{cfg.key}] Downloading model from {cfg.repo_id}...")

    remote_required = []
    remote_optional = []
    for artifact in cfg.required_artifacts:
        if cfg.key == "hindi":
            remote_required.append(f"model_hindi/{artifact}")
        else:
            remote_required.append(artifact)

    for artifact in cfg.optional_artifacts:
        if cfg.key == "hindi":
            remote_optional.append(f"model_hindi/{artifact}")
        else:
            remote_optional.append(artifact)

    for hf_path in remote_required:
        try:
            local_path = hf_hub_download(repo_id=cfg.repo_id, filename=hf_path, repo_type="model")
            shutil.copy2(local_path, os.path.join(cfg.model_dir, os.path.basename(hf_path)))
            print(f"  ✅ {os.path.basename(hf_path)}")
        except Exception as error:
            print(f"  ❌ Error downloading {hf_path}: {error}")
            return False

    for hf_path in remote_optional:
        try:
            local_path = hf_hub_download(repo_id=cfg.repo_id, filename=hf_path, repo_type="model")
            shutil.copy2(local_path, os.path.join(cfg.model_dir, os.path.basename(hf_path)))
            print(f"  ✅ {os.path.basename(hf_path)}")
        except Exception:
            print(f"  ⚠️ Optional file missing: {hf_path}")

    try:
        repo_files = list_repo_files(repo_id=cfg.repo_id, repo_type="model")
    except Exception as error:
        print(f"  ❌ Failed to list repo files: {error}")
        return False

    snapshot_files = [
        file_name for file_name in repo_files
        if file_name.startswith(cfg.snapshot_prefix_remote)
    ]

    if not snapshot_files:
        print("  ❌ No snapshot files found in repository")
        return False

    print(f"\n📥 [{cfg.key}] Downloading snapshot files...")
    for hf_path in snapshot_files:
        try:
            local_path = hf_hub_download(repo_id=cfg.repo_id, filename=hf_path, repo_type="model")
            shutil.copy2(local_path, os.path.join(cfg.model_dir, os.path.basename(hf_path)))
            print(f"  ✅ {os.path.basename(hf_path)}")
        except Exception as error:
            print(f"  ❌ Error downloading {hf_path}: {error}")
            return False

    print(f"\n📥 [{cfg.key}] Downloading code files...")
    for remote_name, local_name in cfg.code_files:
        try:
            local_path = hf_hub_download(repo_id=cfg.repo_id, filename=remote_name, repo_type="model")
            shutil.copy2(local_path, os.path.join(CODE_DIR, local_name))
            print(f"  ✅ {local_name}")
        except Exception as error:
            if "build_charlist" in remote_name.lower():
                print(f"  ⚠️ Optional code file missing: {remote_name}")
            else:
                print(f"  ❌ Error downloading {remote_name}: {error}")
                return False

    return True


def apply_model_fixes(cfg: ModelConfig):
    if cfg.key == "greek":
        replacements = [
            ("../model_sentence/", "model_greek/"),
            ("..\\model_sentence\\", "model_greek\\"),
            ("tv.initialized_value()", "tv"),
        ]
        for file_name in ["Model_Greek.py", "main_Greek.py"]:
            file_path = os.path.join(CODE_DIR, file_name)
            if not os.path.exists(file_path):
                continue
            with open(file_path, "r", encoding="utf-8") as file_obj:
                content = file_obj.read()
            for old_text, new_text in replacements:
                content = content.replace(old_text, new_text)
            with open(file_path, "w", encoding="utf-8") as file_obj:
                file_obj.write(content)

    if cfg.key == "hindi":
        file_path = os.path.join(CODE_DIR, "Model_Hindi_v2.py")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file_obj:
                content = file_obj.read()
            content = content.replace("tv.initialized_value()", "tv")
            with open(file_path, "w", encoding="utf-8") as file_obj:
                file_obj.write(content)

    print(f"✅ [{cfg.key}] Bug fixes applied")


def load_module_from_file(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_model(model_key: str):
    cfg = MODEL_CONFIGS[model_key]
    state = MODEL_STATES[model_key]

    if state["initialized"] and state["model"] is not None:
        return True

    try:
        print(f"\n🔧 [{model_key}] Setting up TensorFlow...")
        tf.keras.backend.clear_session()
        tf.compat.v1.disable_eager_execution()
        os.chdir(CODE_DIR)

        model_file_path = os.path.join(CODE_DIR, cfg.model_file_local)
        pre_file_path = os.path.join(CODE_DIR, cfg.preprocessor_file_local)

        model_module = load_module_from_file(f"model_module_{model_key}", model_file_path)
        pre_module = load_module_from_file(f"pre_module_{model_key}", pre_file_path)

        Model = model_module.Model
        DecoderType = model_module.DecoderType
        preprocess = pre_module.preprocess

        with open(os.path.join(cfg.model_dir, "charList.txt"), "r", encoding="utf-8") as file_obj:
            char_list = file_obj.read()

        latest_snapshot = get_latest_snapshot_prefix(cfg.model_dir)
        if latest_snapshot is None:
            raise FileNotFoundError(f"No snapshot found in {cfg.model_dir}")

        model_signature = inspect.signature(Model.__init__)
        if "restorePath" in model_signature.parameters:
            loaded_model = Model(
                char_list,
                DecoderType.BestPath,
                mustRestore=True,
                restorePath=latest_snapshot,
            )
        else:
            loaded_model = Model(char_list, DecoderType.BestPath, mustRestore=True)

        state["model"] = loaded_model
        state["charList"] = char_list
        state["initialized"] = True
        state["model_class"] = Model
        state["preprocess_fn"] = preprocess
        print(f"✅ [{model_key}] Model restored successfully!")
        return True
    except Exception as error:
        print(f"❌ [{model_key}] Error loading model: {error}")
        import traceback
        traceback.print_exc()
        state["initialized"] = False
        return False


def model_files_exist(cfg: ModelConfig):
    has_charlist = os.path.exists(os.path.join(cfg.model_dir, "charList.txt"))
    has_snapshot = get_latest_snapshot_prefix(cfg.model_dir) is not None
    has_model_file = os.path.exists(os.path.join(CODE_DIR, cfg.model_file_local))
    has_pre_file = os.path.exists(os.path.join(CODE_DIR, cfg.preprocessor_file_local))
    return has_charlist and has_snapshot and has_model_file and has_pre_file


def initialize_model(model_key: str):
    cfg = MODEL_CONFIGS[model_key]
    state = MODEL_STATES[model_key]

    if state["initialized"]:
        return True

    if state["initializing"]:
        print(f"⏳ [{model_key}] Initialization already in progress")
        return False

    state["initializing"] = True

    try:
        print(f"\n🚀 Initializing model: {model_key}")

        if not model_files_exist(cfg):
            print(f"📥 [{model_key}] Model files not found locally, downloading...")
            if not download_model(cfg):
                print(f"❌ [{model_key}] Failed to download model")
                return False
        else:
            print(f"✅ [{model_key}] Model files found locally")

        setup_checkpoint(cfg)
        apply_model_fixes(cfg)
        if not load_model(model_key):
            print(f"❌ [{model_key}] Failed to load model")
            return False

        print(f"✅ [{model_key}] Model initialized successfully")
        return True
    finally:
        state["initializing"] = False


def recognize(image_path: str, model_key: str):
    cfg = MODEL_CONFIGS[model_key]
    state = MODEL_STATES[model_key]

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    if not state["initialized"] or state["model"] is None:
        raise RuntimeError(f"Model '{model_key}' is not initialized")

    img_proc = state["preprocess_fn"](img, state["model_class"].imgSize, dataAugmentation=False)
    batch = np.expand_dims(img_proc, axis=0)

    class _Batch:
        def __init__(self, imgs):
            self.imgs = imgs

    result = state["model"].inferBatch(_Batch(batch))[0]
    print(f"✅ [{cfg.key}] Prediction done")
    return result


# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/predict")
def predict_page():
    return render_template("predict.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/api/upload", methods=["POST"])
def upload_file():
    try:
        model_key = get_model_key(request.form.get("model"))

        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Allowed: " + ", ".join(ALLOWED_EXTENSIONS)}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        print(f"\n📸 Processing: {filename} | model={model_key}")

        if not MODEL_STATES[model_key]["initialized"]:
            if not initialize_model(model_key):
                return jsonify({"error": f"Model '{model_key}' failed to initialize"}), 503

        predicted_text = recognize(filepath, model_key)

        return jsonify(
            {
                "success": True,
                "filename": filename,
                "filepath": url_for("uploaded_file", filename=filename),
                "image_url": url_for("uploaded_file", filename=filename),
                "predicted_text": predicted_text,
                "model": model_key,
            }
        ), 200
    except Exception as error:
        print(f"❌ Upload error: {error}")
        return jsonify({"error": f"Upload failed: {str(error)}"}), 500


@app.route("/api/status")
def status():
    model_key = get_model_key(request.args.get("model"))
    state = MODEL_STATES[model_key]
    return jsonify(
        {
            "model": model_key,
            "model_loaded": state["model"] is not None,
            "charlist_loaded": state["charList"] is not None,
            "initialized": state["initialized"],
            "initializing": state["initializing"],
            "available_models": list(MODEL_CONFIGS.keys()),
        }
    )


@app.route("/api/init-model", methods=["POST"])
def init_model_api():
    model_param = request.form.get("model")
    if model_param is None and request.is_json:
        payload = request.get_json(silent=True) or {}
        model_param = payload.get("model")

    model_key = get_model_key(model_param)
    if MODEL_STATES[model_key]["initialized"]:
        return jsonify({"success": True, "model": model_key, "initialized": True, "already": True}), 200

    ok = initialize_model(model_key)
    if ok:
        return jsonify({"success": True, "model": model_key, "initialized": True}), 200
    return jsonify({"success": False, "model": model_key, "initialized": MODEL_STATES[model_key]["initialized"]}), 503


@app.route("/api/models")
def models_info():
    payload = {}
    for key, cfg in MODEL_CONFIGS.items():
        payload[key] = {
            "display_name": cfg.display_name,
            "repo_id": cfg.repo_id,
            "initialized": MODEL_STATES[key]["initialized"],
        }
    return jsonify({"default": DEFAULT_MODEL_KEY, "models": payload})


@app.before_request
def before_request():
    if request.path in {"/", "/predict", "/api/status", "/api/models", "/api/init-model"}:
        return
    if not MODEL_STATES[DEFAULT_MODEL_KEY]["initialized"]:
        initialize_model(DEFAULT_MODEL_KEY)


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Page not found"}), 404


@app.errorhandler(500)
def server_error(error):
    print(f"❌ Server error: {error}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print("🌐 Starting OCR Web Application...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    for key, cfg in MODEL_CONFIGS.items():
        print(f"📁 {key} model directory: {cfg.model_dir}")
    print(f"🔗 Visit: http://localhost:{port}")
    print("")

    initialize_model(DEFAULT_MODEL_KEY)
    app.run(debug=app_config.DEBUG, host="0.0.0.0", port=port, use_reloader=False)
