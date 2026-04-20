// Upload page functionality
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const dropZone = document.getElementById('dropZone');
const uploadForm = document.getElementById('uploadForm');
const uploadStatus = document.getElementById('uploadStatus');

let modelReady = false;

// Check model status on page load
document.addEventListener('DOMContentLoaded', function() {
    checkModelStatus();
    // Check every 5 seconds while page is open
    setInterval(checkModelStatus, 5000);
});

// Check if model is loaded
async function checkModelStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        if (data.model_loaded && data.charlist_loaded) {
            modelReady = true;
            updateStatusDisplay('ready');
        } else {
            modelReady = false;
            updateStatusDisplay('loading');
        }
    } catch (error) {
        console.error('Error checking model status:', error);
        modelReady = false;
        updateStatusDisplay('checking');
    }
}

function updateStatusDisplay(status) {
    const indicatorDot = document.querySelector('.indicator-dot');
    const indicatorText = document.getElementById('modelIndicatorText');
    
    if (status === 'ready') {
        modelReady = true;
        indicatorDot.classList.remove('loading', 'error');
        indicatorDot.classList.add('ready');
        indicatorText.textContent = 'Model Ready ✓';
        uploadStatus.textContent = '✅ Model loaded and ready! Upload an image to recognize text.';
        uploadStatus.className = 'status success';
        uploadBtn.disabled = false;
    } else if (status === 'loading') {
        modelReady = false;
        indicatorDot.classList.remove('ready', 'error');
        indicatorDot.classList.add('loading');
        indicatorText.textContent = 'Initializing Model...';
        uploadStatus.textContent = '⏳ Model is initializing... This may take a few minutes on first run.';
        uploadStatus.className = 'status uploading';
        uploadBtn.disabled = true;
    } else {
        modelReady = false;
        indicatorDot.classList.remove('ready', 'error');
        indicatorDot.classList.add('loading');
        indicatorText.textContent = 'Checking model...';
        uploadStatus.textContent = '🔄 Checking model status...';
        uploadStatus.className = 'status uploading';
        uploadBtn.disabled = true;
    }
}

// File input change handler
fileInput.addEventListener('change', function() {
    if (this.files.length > 0) {
        if (modelReady) {
            uploadBtn.disabled = false;
            uploadStatus.textContent = `Selected: ${this.files[0].name}`;
            uploadStatus.className = 'status uploading';
        } else {
            uploadBtn.disabled = true;
            uploadStatus.textContent = 'Please wait for the model to load before uploading.';
            uploadStatus.className = 'status uploading';
        }
    }
});

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
    
    if (!modelReady) {
        uploadStatus.textContent = '❌ Model is still initializing. Please wait...';
        uploadStatus.className = 'status error';
        return;
    }
    
    if (fileInput.files.length === 0) {
        uploadStatus.textContent = '❌ Please select a file';
        uploadStatus.className = 'status error';
        return;
    }
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
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
                predicted_text: data.predicted_text
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
