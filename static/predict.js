// Predict page functionality
document.addEventListener('DOMContentLoaded', function() {
    const uploadedData = sessionStorage.getItem('uploadedFile');
    
    if (uploadedData) {
        try {
            const data = JSON.parse(uploadedData);
            displayResults(data);
            sessionStorage.removeItem('uploadedFile'); // Clear data after displaying
        } catch (error) {
            showError('Failed to load prediction results: ' + error.message);
        }
    } else {
        showError('No data available. Please upload an image first.');
    }
});

function displayResults(data) {
    const resultsContainer = document.getElementById('resultsContainer');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const errorContainer = document.getElementById('errorContainer');
    
    // Hide loading and error
    loadingSpinner.style.display = 'none';
    errorContainer.style.display = 'none';
    
    // Display results
    document.getElementById('uploadedImage').src = data.filepath;
    document.getElementById('filename').textContent = `File: ${data.filename}`;
    if (data.model) {
        document.getElementById('modelUsed').textContent = `Model: ${data.model}`;
    } else {
        document.getElementById('modelUsed').textContent = '';
    }
    document.getElementById('predictedText').textContent = data.predicted_text;
    resultsContainer.style.display = 'grid';
}

function showError(message) {
    const loadingSpinner = document.getElementById('loadingSpinner');
    const errorContainer = document.getElementById('errorContainer');
    
    loadingSpinner.style.display = 'none';
    errorContainer.style.display = 'block';
    document.getElementById('errorMessage').textContent = message;
}

// Copy to clipboard functionality
document.getElementById('copyBtn').addEventListener('click', function() {
    const predictedText = document.getElementById('predictedText').textContent;
    
    navigator.clipboard.writeText(predictedText).then(function() {
        const notification = document.getElementById('copyNotification');
        notification.style.display = 'block';
        
        setTimeout(() => {
            notification.style.display = 'none';
        }, 2000);
    }).catch(function(error) {
        alert('Failed to copy text: ' + error);
    });
});
