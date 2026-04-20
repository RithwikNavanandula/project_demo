// Upload page functionality
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const dropZone = document.getElementById('dropZone');
const uploadForm = document.getElementById('uploadForm');
const uploadStatus = document.getElementById('uploadStatus');
const modelSelect = document.getElementById('modelSelect');

let modelReady = false;

// Check model status on page load
document.addEventListener('DOMContentLoaded', function() {
    checkModelStatus();
    // Check every 5 seconds while page is open
    setInterval(checkModelStatus, 5000);
});

if (modelSelect) {
    modelSelect.addEventListener('change', async function() {
        uploadStatus.textContent = `🔄 Switching to ${modelSelect.value} model...`;
        uploadStatus.className = 'status uploading';
        await checkModelStatus();
    });
}

// Check if model is loaded
async function checkModelStatus() {
    try {
        const selectedModel = modelSelect ? modelSelect.value : 'hindi';
        const response = await fetch(`/api/status?model=${encodeURIComponent(selectedModel)}`);
        const data = await response.json();
        
        if (data.model_loaded && data.charlist_loaded) {
            modelReady = true;
            updateStatusDisplay('ready', data.model || selectedModel);
        } else if (data.initializing) {
            modelReady = false;
            updateStatusDisplay('loading', data.model || selectedModel);
        } else {
            modelReady = false;
            updateStatusDisplay('idle', data.model || selectedModel);
        }
    } catch (error) {
        console.error('Error checking model status:', error);
        modelReady = false;
        updateStatusDisplay('checking', modelSelect ? modelSelect.value : 'hindi');
    }
}

function updateStatusDisplay(status, modelName) {
    const indicatorDot = document.querySelector('.indicator-dot');
    const indicatorText = document.getElementById('modelIndicatorText');
    const modelLabel = (modelName || 'hindi').toLowerCase();
    
    if (status === 'ready') {
        modelReady = true;
        indicatorDot.classList.remove('loading', 'error');
        indicatorDot.classList.add('ready');
        indicatorText.textContent = `${modelLabel} model ready ✓`;
        uploadStatus.textContent = `✅ ${modelLabel} model loaded and ready! Upload an image to recognize text.`;
        uploadStatus.className = 'status success';
        uploadBtn.disabled = false;
    } else if (status === 'loading') {
        modelReady = false;
        indicatorDot.classList.remove('ready', 'error');
        indicatorDot.classList.add('loading');
        indicatorText.textContent = `Initializing ${modelLabel} model...`;
        uploadStatus.textContent = `⏳ ${modelLabel} model is initializing... This may take a few minutes on first run.`;
        uploadStatus.className = 'status uploading';
        uploadBtn.disabled = true;
    } else if (status === 'idle') {
        modelReady = false;
        indicatorDot.classList.remove('ready', 'error');
        indicatorDot.classList.add('loading');
        indicatorText.textContent = `${modelLabel} model not initialized`;
        uploadStatus.textContent = `ℹ️ ${modelLabel} model is not initialized yet. Uploading will initialize it.`;
        uploadStatus.className = 'status uploading';
        uploadBtn.disabled = fileInput.files.length === 0;
    } else {
        modelReady = false;
        indicatorDot.classList.remove('ready', 'error');
        indicatorDot.classList.add('loading');
        indicatorText.textContent = `Checking ${modelLabel} model...`;
        uploadStatus.textContent = `🔄 Checking ${modelLabel} model status...`;
        uploadStatus.className = 'status uploading';
        uploadBtn.disabled = true;
    }
}

// File input change handler
fileInput.addEventListener('change', function() {
    if (this.files.length > 0) {
        uploadBtn.disabled = false;
        uploadStatus.textContent = `Selected: ${this.files[0].name}`;
        uploadStatus.className = 'status uploading';
    }
});

async function initializeSelectedModel() {
    const selectedModel = modelSelect ? modelSelect.value : 'hindi';
    const formData = new FormData();
    formData.append('model', selectedModel);

    const response = await fetch('/api/init-model', {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || `Failed to initialize ${selectedModel} model`);
    }
}

// Drag and drop handlers
dropZone.addEventListener('dragover', function(e) {
    e.preventDefault();
    this.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', function() {
    this.classList.remove('drag-over');
});

dropZone.addEventListener('drop', function(e) {
    e.preventDefault();
    this.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        fileInput.dispatchEvent(new Event('change'));
    }
});

// Click on drop zone to open file picker
dropZone.addEventListener('click', function() {
    fileInput.click();
});

// Upload button handler
uploadBtn.addEventListener('click', async function(e) {
    e.preventDefault();
    
    // Check model status before uploading
    await checkModelStatus();
    
    if (fileInput.files.length === 0) {
        uploadStatus.textContent = '❌ Please select a file';
        uploadStatus.className = 'status error';
        return;
    }

    if (!modelReady) {
        const selectedModel = modelSelect ? modelSelect.value : 'hindi';
        uploadBtn.disabled = true;
        uploadStatus.textContent = `⏳ Initializing ${selectedModel} model...`;
        uploadStatus.className = 'status uploading';

        try {
            await initializeSelectedModel();
            await checkModelStatus();
        } catch (error) {
            uploadStatus.textContent = `❌ ${error.message}`;
            uploadStatus.className = 'status error';
            uploadBtn.disabled = false;
            return;
        }

        if (!modelReady) {
            uploadStatus.textContent = `❌ ${selectedModel} model is not ready yet. Please try again.`;
            uploadStatus.className = 'status error';
            uploadBtn.disabled = false;
            return;
        }
    }
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('model', modelSelect ? modelSelect.value : 'hindi');
    
    uploadBtn.disabled = true;
    uploadStatus.textContent = '⏳ Uploading and processing...';
    uploadStatus.className = 'status uploading';
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            // Store data in session storage for the predict page
            sessionStorage.setItem('uploadedFile', JSON.stringify({
                filename: data.filename,
                filepath: data.filepath,
                predicted_text: data.predicted_text,
                model: data.model || (modelSelect ? modelSelect.value : 'hindi')
            }));
            
            uploadStatus.textContent = '✅ Processing complete! Redirecting...';
            uploadStatus.className = 'status success';
            
            // Redirect to predict page after a short delay
            setTimeout(() => {
                window.location.href = '/predict';
            }, 1000);
        } else {
            uploadStatus.textContent = `❌ Error: ${data.error || 'Unknown error'}`;
            uploadStatus.className = 'status error';
            uploadBtn.disabled = false;
        }
    } catch (error) {
        uploadStatus.textContent = `❌ Error: ${error.message}`;
        uploadStatus.className = 'status error';
        uploadBtn.disabled = false;
    }
});
