document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const selectBtn = document.getElementById('selectBtn');
    const previewSection = document.getElementById('previewSection');
    const previewImage = document.getElementById('previewImage');
    const closePreview = document.getElementById('closePreview');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const changeImageBtn = document.getElementById('changeImageBtn');
    const loadingSection = document.getElementById('loadingSection');
    const resultsSection = document.getElementById('resultsSection');
    const resultImage = document.getElementById('resultImage');
    const predictionsDiv = document.getElementById('predictions');
    const treatmentDetails = document.getElementById('treatmentDetails');
    const newAnalysisBtn = document.getElementById('newAnalysisBtn');
    const downloadReportBtn = document.getElementById('downloadReportBtn');

    // API URL (change to your deployed backend URL when live)
    const API_URL = 'http://localhost:8000';
    
    // Pest treatment database
    const treatmentDB = {
        'aphids': {
            organic: 'Spray neem oil or insecticidal soap. Introduce ladybugs.',
            chemical: 'Apply imidacloprid or pyrethroids.',
            prevention: 'Use reflective mulches and avoid over-fertilizing.'
        },
        'beetles': {
            organic: 'Handpick beetles. Use row covers.',
            chemical: 'Apply carbaryl or spinosad.',
            prevention: 'Rotate crops and maintain garden hygiene.'
        },
        'caterpillars': {
            organic: 'Use Bacillus thuringiensis (Bt). Handpick.',
            chemical: 'Apply chlorantraniliprole.',
            prevention: 'Use pheromone traps.'
        },
        'grasshoppers': {
            organic: 'Apply neem oil. Use nosema locustae.',
            chemical: 'Use carbaryl bait.',
            prevention: 'Keep vegetation short around fields.'
        },
        'spider_mites': {
            organic: 'Spray water to dislodge. Use predatory mites.',
            chemical: 'Apply miticides like abamectin.',
            prevention: 'Maintain humidity and avoid drought stress.'
        }
    };

    // Default treatment for unknown pests
    const defaultTreatment = {
        organic: 'Consult local agricultural extension office.',
        chemical: 'Consult with pest control professional.',
        prevention: 'Regular crop monitoring and good farming practices.'
    };

    // Upload area click
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // Select button click
    selectBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#667eea';
        uploadArea.style.background = '#f7fafc';
    });

    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#cbd5e0';
        uploadArea.style.background = 'white';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#cbd5e0';
        uploadArea.style.background = 'white';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Close preview
    closePreview.addEventListener('click', () => {
        previewSection.style.display = 'none';
        uploadArea.style.display = 'block';
        fileInput.value = '';
    });

    // Change image
    changeImageBtn.addEventListener('click', () => {
        fileInput.click();
    });

    // Analyze button
    analyzeBtn.addEventListener('click', analyzeImage);

    // New analysis button
    newAnalysisBtn.addEventListener('click', () => {
        resultsSection.style.display = 'none';
        uploadArea.style.display = 'block';
        previewSection.style.display = 'none';
        fileInput.value = '';
    });

    // Download report
    downloadReportBtn.addEventListener('click', downloadReport);

    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }

        // Validate file size (10MB max)
        if (file.size > 10 * 1024 * 1024) {
            alert('File size must be less than 10MB.');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            resultImage.src = e.target.result;
            uploadArea.style.display = 'none';
            previewSection.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    async function analyzeImage() {
        const file = fileInput.files[0];
        if (!file) return;

        // Show loading
        previewSection.style.display = 'none';
        loadingSection.style.display = 'block';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Analysis failed');
            }

            const data = await response.json();
            displayResults(data.predictions);
        } catch (error) {
            alert('Error analyzing image. Using mock predictions for demo.');
            // Mock predictions for demo
            const mockPredictions = [
                { class: 'aphids', confidence: 0.92 },
                { class: 'beetles', confidence: 0.78 },
                { class: 'caterpillars', confidence: 0.65 }
            ];
            displayResults(mockPredictions);
        }
    }

    function displayResults(predictions) {
        loadingSection.style.display = 'none';
        
        // Display predictions
        predictionsDiv.innerHTML = '';
        predictions.forEach(pred => {
            const confidencePercent = (pred.confidence * 100).toFixed(1);
            const predElement = document.createElement('div');
            predElement.className = 'prediction-item';
            predElement.innerHTML = `
                <span class="prediction-class">${pred.class}</span>
                <span class="prediction-confidence">${confidencePercent}%</span>
            `;
            predictionsDiv.appendChild(predElement);
        });

        // Display treatment info for top prediction
        const topPest = predictions[0].class.toLowerCase();
        const treatment = treatmentDB[topPest] || defaultTreatment;
        
        treatmentDetails.innerHTML = `
            <div style="margin-bottom: 15px;">
                <strong>🌱 Organic Treatment:</strong><br>
                ${treatment.organic}
            </div>
            <div style="margin-bottom: 15px;">
                <strong>🧪 Chemical Treatment:</strong><br>
                ${treatment.chemical}
            </div>
            <div>
                <strong>🛡️ Prevention:</strong><br>
                ${treatment.prevention}
            </div>
        `;

        resultsSection.style.display = 'block';
    }

    function downloadReport() {
        const predictions = document.querySelectorAll('.prediction-item');
        const date = new Date().toLocaleString();
        
        let report = 'AI PEST DETECTION REPORT\n';
        report += '='.repeat(40) + '\n';
        report += `Date: ${date}\n\n`;
        report += 'DETECTED PESTS:\n';
        
        predictions.forEach((pred, index) => {
            const pestClass = pred.querySelector('.prediction-class').textContent;
            const confidence = pred.querySelector('.prediction-confidence').textContent;
            report += `${index + 1}. ${pestClass}: ${confidence}\n`;
        });
        
        report += '\nRECOMMENDED TREATMENT:\n';
        report += treatmentDetails.innerText;
        
        const blob = new Blob([report], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `pest-detection-report-${Date.now()}.txt`;
        a.click();
        window.URL.revokeObjectURL(url);
    }

    // Check API health on load
    fetch(`${API_URL}/health`)
        .then(response => response.json())
        .then(data => console.log('API Status:', data))
        .catch(error => console.warn('API not reachable - using mock mode'));
});

// Theme toggle functionality
const themeToggle = document.getElementById('themeToggle');
const themeIcon = document.getElementById('themeIcon');
const themeText = document.getElementById('themeText');

// Check for saved theme preference
const savedTheme = localStorage.getItem('theme') || 'light';
document.documentElement.setAttribute('data-theme', savedTheme);
updateThemeButton(savedTheme);

themeToggle.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeButton(newTheme);
});

function updateThemeButton(theme) {
    if (theme === 'dark') {
        themeIcon.textContent = '☀️';
        themeText.textContent = 'Light Mode';
    } else {
        themeIcon.textContent = '🌙';
        themeText.textContent = 'Dark Mode';
    }
}