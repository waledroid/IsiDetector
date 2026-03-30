const translations = {
    en: {
        "title": "ISITEC visionAI Platform",
        "nav_live": "Live Inference",
        "nav_analytics": "Analytics",
        "nav_models": "Models",
        "nav_settings": "Settings",
        "nav_about": "About",
        "stream_config": "Stream Configuration",
        "modes": "Modes",
        "input_source": "Input Source",
        "src_image": "Image",
        "src_video": "Video",
        "src_rtsp": "RTSP URL",
        "src_camera": "Camera",
        "choose_file": "Choose File",
        "drag_drop": "or drag and drop here",
        "start_tracking": "Start Tracking",
        "stop": "Stop",
        "analytics_overview": "Analytics & Logging Overview",
        "analytics_soon": "Historical logs are stored in the /logs directory and parsed in the live dashboard natively.",
        "model_management": "Model Management",
        "model_soon": "Model Directory Viewer Coming Soon.",
        "system_settings": "System Settings",
        "about_title": "About ISITEC visionAI",
        "about_desc": "Industrial Object Detection and Tracking Platform",
        "version": "Version 1.0.0-beta",
        "copyright_footer": "&copy; Copyright Isitec International 2026",
        "stat_cartons": "Cartons",
        "stat_polybags": "Polybags",
        "stat_last": "Last Detected",
        "stat_none": "None",
        "chart_title": "Detection Analytics",
        "filter_live": "Live",
        "filter_24h": "24h",
        "filter_7d": "7 Days",
        "filter_30d": "30 Days"
    },
    fr: {
        "title": "Plateforme ISITEC visionAI",
        "nav_live": "Inférence en direct",
        "nav_analytics": "Analytique",
        "nav_models": "Modèles",
        "nav_settings": "Paramètres",
        "nav_about": "À propos",
        "stream_config": "Configuration du flux",
        "modes": "Modes",
        "input_source": "Source d'entrée",
        "src_image": "Image",
        "src_video": "Vidéo",
        "src_rtsp": "URL RTSP",
        "src_camera": "Caméra",
        "choose_file": "Choisir un fichier",
        "drag_drop": "ou glissez et déposez ici",
        "start_tracking": "Démarrer le suivi",
        "stop": "Arrêter",
        "analytics_overview": "Aperçu de l'analytique et des journaux",
        "analytics_soon": "Les journaux historiques sont stockés dans /logs et analysés dans le tableau de bord.",
        "model_management": "Gestion des modèles",
        "model_soon": "Visionneur de répertoire de modèles à venir.",
        "system_settings": "Paramètres du système",
        "about_title": "À propos d'ISITEC visionAI",
        "about_desc": "Plateforme industrielle de détection et de suivi d'objets",
        "version": "Version 1.0.0-bêta",
        "copyright_footer": "&copy; Droit d'auteur Isitec International 2026",
        "stat_cartons": "Cartons",
        "stat_polybags": "Sachets (Polybags)",
        "stat_last": "Dernier détecté",
        "stat_none": "Aucun",
        "chart_title": "Analyse de détection",
        "filter_live": "En direct",
        "filter_24h": "24h",
        "filter_7d": "7 Jours",
        "filter_30d": "30 Jours"
    }
};

const msgTrans = {
    en: {
        "msg_uploading": "Uploading file...",
        "msg_select_file": "Please select a file.",
        "msg_network_err": "Network error.",
        "msg_starting": "Starting inference stream...",
        "msg_enter_url": "Please enter a valid source URL/ID.",
        "placeholder_rtsp": "Enter RTSP URL (e.g. rtsp://192.168.1.100:554/stream)",
        "placeholder_cam": "Enter Camera ID (e.g. 0 or 1)"
    },
    fr: {
        "msg_uploading": "Téléchargement en cours...",
        "msg_select_file": "Veuillez sélectionner un fichier.",
        "msg_network_err": "Erreur réseau.",
        "msg_starting": "Démarrage automatique du flux d'inférence...",
        "msg_enter_url": "Veuillez entrer une URL / ID de source valide.",
        "placeholder_rtsp": "Entrez l'URL RTSP (ex. rtsp://192.168.1.100:554/stream)",
        "placeholder_cam": "Entrez l'ID de la caméra (ex. 0 ou 1)"
    }
};

document.addEventListener('DOMContentLoaded', () => {
    
    // --- I18N Translation Logic ---
    let currentLang = 'en';
    const btnEn = document.getElementById('btn-lang-en');
    const btnFr = document.getElementById('btn-lang-fr');
    
    function setLanguage(lang) {
        currentLang = lang;
        if (lang === 'fr') {
            btnFr.classList.add('active');
            btnEn.classList.remove('active');
        } else {
            btnEn.classList.add('active');
            btnFr.classList.remove('active');
        }

        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            if (translations[lang] && translations[lang][key]) {
                el.innerHTML = translations[lang][key]; 
            }
        });

        const sourceText = document.getElementById('sourceText');
        if (currentSourceType === 'rtsp') {
            sourceText.placeholder = msgTrans[lang].placeholder_rtsp;
        } else if (currentSourceType === 'camera') {
            sourceText.placeholder = msgTrans[lang].placeholder_cam;
        }

        if (document.getElementById('sourceFile').files.length === 0) {
            document.querySelector('.file-msg').textContent = translations[lang].drag_drop;
        }

        fetch('/api/language', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ language: lang })
        }).catch(e => console.error("Could not notify backend"));
    }

    btnEn.addEventListener('click', () => setLanguage('en'));
    btnFr.addEventListener('click', () => setLanguage('fr'));

    // --- Navigation Panel Switcher ---
    const navBtns = document.querySelectorAll('.nav-btn');
    const panels = document.querySelectorAll('.panel');

    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            navBtns.forEach(b => b.classList.remove('active'));
            panels.forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // --- Toast Strip ---
    const messageStrip = document.getElementById('message');
    let messageTimeout;

    function showMessage(text, type = 'info') {
        messageStrip.textContent = text;
        messageStrip.className = `message-strip ${type}`;
        clearTimeout(messageTimeout);
        messageTimeout = setTimeout(() => {
            messageStrip.classList.add('hidden');
        }, 3500);
    }

    // --- Chart.js Initialization ---
    let detectionChart = null;
    let currentChartPeriod = 'live';
    
    const ctx = document.getElementById('analyticsChart').getContext('2d');
    detectionChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Count',
                data: [],
                backgroundColor: [
                    'rgba(63, 81, 181, 0.7)',
                    'rgba(255, 64, 129, 0.7)',
                    'rgba(76, 175, 80, 0.7)'
                ],
                borderColor: [
                    'rgba(63, 81, 181, 1)',
                    'rgba(255, 64, 129, 1)',
                    'rgba(76, 175, 80, 1)'
                ],
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { precision: 0 }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });

    // Chart Filter Buttons Logic
    const filterBtns = document.querySelectorAll('.filter-btn');
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentChartPeriod = btn.getAttribute('data-period');
            updateChartData(currentChartPeriod);
        });
    });

    async function updateChartData(period) {
        try {
            const res = await fetch(`/api/chart?period=${period}`);
            const data = await res.json();
            
            if (data.status === 'success') {
                const labels = Object.keys(data.data);
                const values = Object.values(data.data);
                
                detectionChart.data.labels = labels.map(l => l.toUpperCase());
                detectionChart.data.datasets[0].data = values;
                detectionChart.update();
            }
        } catch (e) {
            console.error("Error fetching chart data", e);
        }
    }


    // --- Stats Polling Logic ---
    let statsInterval = null;
    const statCartons = document.getElementById('statCartons');
    const statPolybags = document.getElementById('statPolybags');
    const statLast = document.getElementById('statLast');
    
    async function fetchStats() {
        try {
            const res = await fetch('/api/stats');
            const data = await res.json();
            
            if (data.counts) {
                let cartonCount = 0;
                let polybagCount = 0;
                for (let key in data.counts) {
                    if (key.toLowerCase().includes('carton')) cartonCount += data.counts[key];
                    if (key.toLowerCase().includes('polybag') || key.toLowerCase().includes('bag')) polybagCount += data.counts[key];
                }
                statCartons.textContent = cartonCount;
                statPolybags.textContent = polybagCount;
                
                // Live chart update
                if (currentChartPeriod === 'live') {
                    const labels = Object.keys(data.counts);
                    const values = Object.values(data.counts);
                    detectionChart.data.labels = labels.map(l => l.toUpperCase());
                    detectionChart.data.datasets[0].data = values;
                    detectionChart.update();
                }
            }
            if (data.last_detected) {
                statLast.textContent = `${data.last_detected.class.toUpperCase()} - ${data.last_detected.time}`;
                statLast.removeAttribute('data-i18n');
            }
            if (!data.is_running && statsInterval) {
                clearInterval(statsInterval);
                statsInterval = null;
            }
        } catch (e) {
            console.error("Failed to fetch stats");
        }
    }

    function startStatsPolling() {
        if (statsInterval) clearInterval(statsInterval);
        statsInterval = setInterval(fetchStats, 1000);
    }

    // Load initial empty chart
    updateChartData('live');

    // --- Source Selection Logic ---
    const sourceBtns = document.querySelectorAll('.source-btn');
    const fileDropArea = document.getElementById('fileDropArea');
    const sourceFile = document.getElementById('sourceFile');
    const sourceText = document.getElementById('sourceText');
    const fileMsg = document.querySelector('.file-msg');
    
    let currentSourceType = 'image';
    let uploadedFilePath = '';

    sourceBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            sourceBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentSourceType = btn.getAttribute('data-type');
            
            if (currentSourceType === 'image') {
                fileDropArea.style.display = 'flex';
                sourceText.style.display = 'none';
                sourceFile.setAttribute('accept', 'image/*');
                fileMsg.textContent = translations[currentLang].drag_drop;
            } else if (currentSourceType === 'video') {
                fileDropArea.style.display = 'flex';
                sourceText.style.display = 'none';
                sourceFile.setAttribute('accept', 'video/mp4,video/x-m4v,video/*');
                fileMsg.textContent = translations[currentLang].drag_drop;
            } else if (currentSourceType === 'rtsp') {
                fileDropArea.style.display = 'none';
                sourceText.style.display = 'block';
                sourceText.placeholder = msgTrans[currentLang].placeholder_rtsp;
                sourceText.value = '';
            } else if (currentSourceType === 'camera') {
                fileDropArea.style.display = 'none';
                sourceText.style.display = 'block';
                sourceText.placeholder = msgTrans[currentLang].placeholder_cam;
                sourceText.value = '0';
            }
        });
    });

    sourceFile.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            fileMsg.textContent = e.target.files[0].name;
            uploadedFilePath = ''; // Reset, needs upload
        }
    });

    // --- API Commands ---
    const btnStart = document.getElementById('btnStart');
    const btnStop = document.getElementById('btnStop');
    const videoStream = document.getElementById('videoStream');
    const videoCanvas = document.getElementById('videoCanvas');
    const vCtx = videoCanvas.getContext('2d');

    function renderCanvas() {
        if (videoStream.complete && videoStream.naturalWidth > 0) {
            // Dynamically adjust canvas to match stream resolution
            if (videoCanvas.width !== videoStream.naturalWidth) {
                videoCanvas.width = videoStream.naturalWidth;
                videoCanvas.height = videoStream.naturalHeight;
            }
            vCtx.drawImage(videoStream, 0, 0, videoCanvas.width, videoCanvas.height);
        }
        requestAnimationFrame(renderCanvas);
    }
    renderCanvas();

    btnStart.addEventListener('click', async () => {
        const model_type = document.getElementById('modelSelect').value;
        const weights = ''; // Backend default

        let finalSource = '';

        if (currentSourceType === 'image' || currentSourceType === 'video') {
            if (sourceFile.files.length === 0 && !uploadedFilePath) {
                showMessage(msgTrans[currentLang].msg_select_file, "error");
                return;
            }
            if (!uploadedFilePath && sourceFile.files.length > 0) {
                showMessage(msgTrans[currentLang].msg_uploading, "info");
                btnStart.disabled = true;
                const formData = new FormData();
                formData.append('file', sourceFile.files[0]);
                
                try {
                    const uploadRes = await fetch('/api/upload', {
                        method: 'POST',
                        body: formData
                    });
                    const uploadData = await uploadRes.json();
                    if (uploadRes.ok) {
                        uploadedFilePath = uploadData.filepath;
                    } else {
                        showMessage(uploadData.message, "error");
                        btnStart.disabled = false;
                        return;
                    }
                } catch (e) {
                    showMessage(msgTrans[currentLang].msg_network_err, "error");
                    btnStart.disabled = false;
                    return;
                }
            }
            finalSource = uploadedFilePath;
        } else {
            finalSource = sourceText.value;
            if (!finalSource) {
                showMessage(msgTrans[currentLang].msg_enter_url, "error");
                return;
            }
        }

        btnStart.disabled = true;
        showMessage(msgTrans[currentLang].msg_starting, "info");

        try {
            const res = await fetch('/api/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source: finalSource, weights, model_type })
            });
            const data = await res.json();
            
            if (res.ok) {
                showMessage(data.message, "success");
                videoStream.src = `/video_feed?t=${new Date().getTime()}`;
                
                // Start tracking stats
                statCartons.textContent = "0";
                statPolybags.textContent = "0";
                statLast.textContent = translations[currentLang].stat_none;
                startStatsPolling();
            } else {
                showMessage(data.message, "error");
            }
        } catch (err) {
            showMessage(msgTrans[currentLang].msg_network_err, "error");
        } finally {
            btnStart.disabled = false;
        }
    });

    btnStop.addEventListener('click', async () => {
        btnStop.disabled = true;
        try {
            const res = await fetch('/api/stop', { method: 'POST' });
            const data = await res.json();
            
            if (res.ok) {
                showMessage(data.message, "info");
                if (statsInterval) clearInterval(statsInterval);
            } else {
                showMessage(data.message, "error");
            }
        } catch (err) {
            showMessage(msgTrans[currentLang].msg_network_err, "error");
        } finally {
            btnStop.disabled = false;
        }
    });
});
