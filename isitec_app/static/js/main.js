const translations = {
    en: {
        "title": "ISITEC visionAI Platform",
        "nav_live": "Live Inference",
        "nav_analytics": "Analytics",
        "nav_performance": "Performance",
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
        "filter_30d": "30 Days",
        "nav_performance": "Performance",
        "perf_title": "Performance Monitoring",
        "perf_subtitle": "Real-time system and model metrics. Start a stream to populate.",
        "pg_session": "Session Health",
        "pg_hardware": "Hardware",
        "pg_throughput": "Throughput",
        "pg_detection": "Detection Quality",
        "pg_tracking": "Tracking",
        "pg_counting": "Counting Rate",
        "pg_sessions": "Session Comparison (Last 5)",
        "sess_date": "Date", "sess_model": "Model", "sess_duration": "Duration",
        "sess_fps": "FPS", "sess_conf": "Avg Conf", "sess_idratio": "ID Ratio",
        "sess_counts": "Counts",
        "perf_empty": "Start a stream to see live metrics"
    },
    fr: {
        "title": "Plateforme ISITEC visionAI",
        "nav_live": "Inférence en direct",
        "nav_analytics": "Analytique",
        "nav_performance": "Performance",
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
        "filter_30d": "30 Jours",
        "perf_title": "Surveillance des performances",
        "perf_subtitle": "Métriques système et modèle en temps réel. Démarrez un flux pour remplir.",
        "pg_session": "État de la session",
        "pg_hardware": "Matériel",
        "pg_throughput": "Débit",
        "pg_detection": "Qualité de détection",
        "pg_tracking": "Suivi",
        "pg_counting": "Taux de comptage",
        "pg_sessions": "Comparaison des sessions (5 dernières)",
        "sess_date": "Date", "sess_model": "Modèle", "sess_duration": "Durée",
        "sess_fps": "IPS", "sess_conf": "Conf. moy.", "sess_idratio": "Ratio ID",
        "sess_counts": "Comptages",
        "perf_empty": "Démarrez un flux pour voir les métriques en direct"
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
            // Immediately fetch perf data when Performance tab is opened
            if (targetId === 'section-performance') fetchPerformance();
            // Fetch session history when Analytics tab is opened
            if (targetId === 'section-analytics') fetchSessionsForAnalytics();
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
                detectionChart.update('none');  // data-only update, no animation overhead
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

                // Live chart update — 'none' skips animation, avoids full re-render cost
                if (currentChartPeriod === 'live') {
                    const labels = Object.keys(data.counts);
                    const values = Object.values(data.counts);
                    detectionChart.data.labels = labels.map(l => l.toUpperCase());
                    detectionChart.data.datasets[0].data = values;
                    detectionChart.update('none');
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
        statsInterval = setInterval(fetchStats, 2000);  // 2s — halves API calls, no UX impact
        startPerfPolling();
    }

    // Load initial empty chart
    updateChartData('live');

    // On page load, check if an inference session is already running and restore
    // the UI silently — avoids resetting counters when the user refreshes mid-session.
    async function restoreSession() {
        try {
            const res = await fetch('/api/stats');
            const data = await res.json();

            if (data.is_running) {
                // Reconnect the MJPEG stream
                videoStream.src = `/video_feed?t=${new Date().getTime()}`;

                // Restore counter display
                if (data.counts) {
                    let cartonCount = 0, polybagCount = 0;
                    for (let key in data.counts) {
                        if (key.toLowerCase().includes('carton')) cartonCount += data.counts[key];
                        if (key.toLowerCase().includes('polybag') || key.toLowerCase().includes('bag')) polybagCount += data.counts[key];
                    }
                    statCartons.textContent = cartonCount;
                    statPolybags.textContent = polybagCount;

                    // Restore live chart
                    detectionChart.data.labels = Object.keys(data.counts).map(l => l.toUpperCase());
                    detectionChart.data.datasets[0].data = Object.values(data.counts);
                    detectionChart.update('none');
                }

                // Restore last-detected label
                if (data.last_detected) {
                    statLast.textContent = `${data.last_detected.class.toUpperCase()} - ${data.last_detected.time}`;
                    statLast.removeAttribute('data-i18n');
                }

                // Resume polling without calling /api/start (which would reset counters)
                startStatsPolling();
                fetchPerformance();
            }
        } catch (e) {
            console.error("Session restore check failed:", e);
        }
    }
    restoreSession();

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
        
        let weights = '';
        let imgsz = null;
        let conf = null;
        
        // Grab values from localStorage if they've been configured in Settings
        if (model_type === 'yolo') {
            weights = localStorage.getItem('isitec_yolo_weights') || '';
            imgsz = localStorage.getItem('isitec_yolo_imgsz');
            conf = localStorage.getItem('isitec_yolo_conf');
        } else if (model_type === 'Detr') { // RF-DETR dropdown value
            weights = localStorage.getItem('isitec_detr_weights') || '';
            imgsz = localStorage.getItem('isitec_detr_imgsz');
            conf = localStorage.getItem('isitec_detr_conf');
        }

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
            const payload = { source: finalSource, weights, model_type };
            if (imgsz) payload.imgsz = parseInt(imgsz);
            if (conf) payload.conf = parseFloat(conf);

            const res = await fetch('/api/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
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
                stopPerfPolling();
            } else {
                showMessage(data.message, "error");
            }
        } catch (err) {
            showMessage(msgTrans[currentLang].msg_network_err, "error");
        } finally {
            btnStop.disabled = false;
        }
    });

    // ── Performance Panel Polling ────────────────────────────────────────────
    let perfInterval = null;

    function startPerfPolling() {
        stopPerfPolling();
        perfInterval = setInterval(fetchPerformance, 2000);
    }
    function stopPerfPolling() {
        if (perfInterval) { clearInterval(perfInterval); perfInterval = null; }
    }

    async function fetchPerformance() {
        // Fetch when Performance OR Analytics tab is visible
        const perfVisible = document.getElementById('section-performance')?.classList.contains('active');
        const analyticsVisible = document.getElementById('section-analytics')?.classList.contains('active');
        if (!perfVisible && !analyticsVisible) return;
        try {
            const res = await fetch('/api/performance');
            const d = await res.json();

            // Show/hide empty-state banner
            const banner = document.getElementById('perfEmptyBanner');
            if (banner) {
                const isRunning = d.session && d.session.is_running;
                banner.classList.toggle('hidden', isRunning);
            }

            if (perfVisible) {
                updatePerfGroup('session',    d.session,    buildSessionRows);
                updatePerfGroup('hardware',   d.hardware,   buildHardwareRows);
                updatePerfGroup('throughput',  d.throughput,  buildThroughputRows);
                updatePerfGroup('detection',   d.detection,   buildDetectionRows);
                updatePerfGroup('tracking',    d.tracking,    buildTrackingRows);
                updatePerfGroup('counting',    d.counting,    buildCountingRows);
            }
            // Always update sessions table (now in Analytics)
            updateSessionsTable(d.sessions);
        } catch (e) {
            console.error('Performance fetch failed', e);
        }
    }

    function updatePerfGroup(name, data, buildFn) {
        const group   = document.getElementById(`pg-${name}`);
        const dot     = document.getElementById(`dot-${name}`);
        const metrics = document.getElementById(`pm-${name}`);
        if (!group || !dot || !metrics) return;
        const status = (data && data.status) || 'green';
        group.className = `perf-group status-${status}`;
        dot.className   = `pg-dot ${status}`;
        metrics.innerHTML = buildFn(data || {});
    }

    // Formatting helpers
    function pmRow(label, valueHtml) {
        return `<div class="pm-row"><span class="pm-label">${label}</span><span class="pm-value">${valueHtml}</span></div>`;
    }
    function fmt(val, unit, dec) {
        if (val == null) return '<span class="pm-na">—</span>';
        return Number(val).toFixed(dec !== undefined ? dec : 1) + (unit || '');
    }
    function fmtPct(val) { return fmt(val, '%', 0); }

    // ── Build functions for each metric group ────────────────────────────────

    function buildSessionRows(d) {
        return pmRow('Uptime',       d.uptime_fmt || '<span class="pm-na">—</span>')
             + pmRow('Status',       d.is_running ? '🟢 LIVE' : '⏹ Idle')
             + pmRow('Errors',       fmt(d.error_count, '', 0))
             + pmRow('CUDA OOM',     fmt(d.cuda_oom_count, '', 0))
             + pmRow('Heartbeat',    d.heartbeat_age_s != null ? fmt(d.heartbeat_age_s, 's', 0) : '<span class="pm-na">—</span>');
    }

    function buildHardwareRows(d) {
        let html = '';
        // VRAM progress bar
        if (d.vram_used_mb != null && d.vram_total_mb != null) {
            const pct = Math.round(d.vram_pct || 0);
            const barClass = pct > 85 ? 'bar-red' : pct > 70 ? 'bar-yellow' : 'bar-green';
            html += `<div class="pm-row" style="flex-direction:column; gap:4px;">
                <div style="display:flex; justify-content:space-between; width:100%;">
                    <span class="pm-label">VRAM</span>
                    <span class="pm-value">${(d.vram_used_mb/1024).toFixed(1)} / ${(d.vram_total_mb/1024).toFixed(1)} GB</span>
                </div>
                <div class="pm-bar-wrap">
                    <div class="pm-bar-track"><div class="pm-bar-fill ${barClass}" style="width:${pct}%"></div></div>
                    <span class="pm-bar-label">${pct}%</span>
                </div>
            </div>`;
        } else {
            html += pmRow('VRAM', '<span class="pm-na">—</span>');
        }
        // GPU Util progress bar
        if (d.gpu_util_pct != null) {
            const pct = Math.round(d.gpu_util_pct);
            const barClass = pct > 90 ? 'bar-red' : pct > 80 ? 'bar-yellow' : 'bar-green';
            html += `<div class="pm-row" style="flex-direction:column; gap:4px;">
                <div style="display:flex; justify-content:space-between; width:100%;">
                    <span class="pm-label">GPU Util</span>
                    <span class="pm-value">${pct}%</span>
                </div>
                <div class="pm-bar-wrap">
                    <div class="pm-bar-track"><div class="pm-bar-fill ${barClass}" style="width:${pct}%"></div></div>
                    <span class="pm-bar-label">${pct}%</span>
                </div>
            </div>`;
        } else {
            html += pmRow('GPU Util', '<span class="pm-na">—</span>');
        }
        html += pmRow('GPU Temp', d.gpu_temp_c != null ? fmt(d.gpu_temp_c, '°C', 0) : '<span class="pm-na">—</span>');
        
        // RAM progress bar
        if (d.ram_used_mb != null && d.ram_total_mb != null) {
            const pct = Math.round(d.ram_pct || 0);
            const barClass = pct > 90 ? 'bar-red' : pct > 80 ? 'bar-yellow' : 'bar-green';
            html += `<div class="pm-row" style="flex-direction:column; gap:4px;">
                <div style="display:flex; justify-content:space-between; width:100%;">
                    <span class="pm-label">System RAM</span>
                    <span class="pm-value">${(d.ram_used_mb/1024).toFixed(1)} / ${(d.ram_total_mb/1024).toFixed(1)} GB</span>
                </div>
                <div class="pm-bar-wrap">
                    <div class="pm-bar-track"><div class="pm-bar-fill ${barClass}" style="width:${pct}%"></div></div>
                    <span class="pm-bar-label">${pct}%</span>
                </div>
            </div>`;
        } else {
            html += pmRow('System RAM', '<span class="pm-na">—</span>');
        }
        
        return html;
    }

    function buildThroughputRows(d) {
        return pmRow('FPS',         fmt(d.fps, '', 1))
             + pmRow('Latency',     fmt(d.latency_ms, 'ms', 1))
             + pmRow('Frame Drops', fmt(d.frame_drops, '', 0))
             + pmRow('Forward',     fmt(d.forward_ms, 'ms', 1))
             + pmRow('Tracker',     fmt(d.tracker_ms, 'ms', 1));
    }

    function buildDetectionRows(d) {
        return pmRow('Avg Confidence', fmt(d.avg_confidence, '', 2))
             + pmRow('Conf σ',         fmt(d.conf_std, '', 3))
             + pmRow('Low-Conf Rate',  d.low_conf_rate != null ? (d.low_conf_rate * 100).toFixed(1) + '%' : '<span class="pm-na">—</span>')
             + pmRow('Dets / Frame',   fmt(d.avg_detections, '', 1))
             + pmRow('Mask Coverage',  fmt(d.mask_coverage, '', 2));
    }

    function buildTrackingRows(d) {
        return pmRow('Unique IDs',     fmt(d.total_unique_ids, '', 0))
             + pmRow('Total Crossings', fmt(d.total_crossings, '', 0))
             + pmRow('ID Ratio',        fmt(d.id_ratio, '', 2));
    }

    function buildCountingRows(d) {
        const totals  = d.totals || {};
        const rates   = d.rate_per_hour || {};
        const keys = Object.keys(totals);
        if (keys.length === 0) return '<div class="pm-row"><span class="pm-label">No data</span><span class="pm-value pm-na">—</span></div>';
        return keys.map(k => {
            const rateStr = rates[k] != null ? ` (${rates[k]}/hr)` : '';
            return pmRow(k.charAt(0).toUpperCase() + k.slice(1), `${totals[k]}${rateStr}`);
        }).join('');
    }

    // ── Session Comparison Table ─────────────────────────────────────────────
    function updateSessionsTable(sessions) {
        const tbody = document.getElementById('sessionsBody');
        if (!tbody) return;
        if (!sessions || sessions.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; color: var(--text-light); padding: 24px;">No sessions recorded yet.</td></tr>';
            return;
        }
        tbody.innerHTML = sessions.map(s => {
            const modelClass = (s.model || '').toLowerCase().includes('rfdetr') ? 'sess-model-rfdetr' : 'sess-model-yolo';
            const countsStr  = s.counts ? Object.entries(s.counts).map(([k,v]) => `${k}: ${v}`).join(', ') : '—';
            return `<tr>
                <td>${s.date || '—'}</td>
                <td class="${modelClass}">${(s.model || '—').toUpperCase()}</td>
                <td>${s.duration_h != null ? s.duration_h.toFixed(1) + 'h' : '—'}</td>
                <td>${s.fps != null ? s.fps.toFixed(1) : '—'}</td>
                <td>${s.avg_confidence != null ? s.avg_confidence.toFixed(3) : '—'}</td>
                <td>${s.id_ratio != null ? s.id_ratio.toFixed(2) : '—'}</td>
                <td>${countsStr}</td>
            </tr>`;
        }).join('');
    }
    // ── Fetch sessions for Analytics tab ─────────────────────────────────────
    async function fetchSessionsForAnalytics() {
        try {
            const res = await fetch('/api/performance');
            const d = await res.json();
            updateSessionsTable(d.sessions);
        } catch (e) {
            console.error('Sessions fetch failed', e);
        }
    }

    // Always start perf polling on load. It silently bails if the tabs aren't active.
    startPerfPolling();

    // ── Settings Panel Logic ────────────────────────────────────────────────
    
    // Bind slider values to display spans
    ['yolo_imgsz', 'yolo_conf', 'detr_imgsz', 'detr_conf'].forEach(id => {
        const slider = document.getElementById(`set_${id}`);
        const valSpan = document.getElementById(`val_${id}`);
        if (slider && valSpan) {
            slider.addEventListener('input', (e) => {
                valSpan.textContent = id.includes('imgsz') ? `${e.target.value}px` : parseFloat(e.target.value).toFixed(2);
            });
        }
    });

    async function loadSettings() {
        // Load sliders from localStorage
        ['yolo_imgsz', 'yolo_conf', 'detr_imgsz', 'detr_conf'].forEach(id => {
            const saved = localStorage.getItem(`isitec_${id}`);
            if (saved) {
                const slider = document.getElementById(`set_${id}`);
                const valSpan = document.getElementById(`val_${id}`);
                if (slider && valSpan) {
                    slider.value = saved;
                    valSpan.textContent = id.includes('imgsz') ? `${saved}px` : parseFloat(saved).toFixed(2);
                }
            }
        });

        // Fetch models from API
        try {
            const res = await fetch('/api/models');
            const data = await res.json();
            if (data.status === 'success') {
                const yoloSelect = document.getElementById('set_yolo_weights');
                const detrSelect = document.getElementById('set_detr_weights');
                
                // Clear loading options
                yoloSelect.innerHTML = '<option value="">Default weights</option>';
                detrSelect.innerHTML = '<option value="">Default weights</option>';

                data.yolo_models.forEach(m => {
                    yoloSelect.innerHTML += `<option value="${m.path}">${m.name} (${m.path})</option>`;
                });
                data.rfdetr_models.forEach(m => {
                    detrSelect.innerHTML += `<option value="${m.path}">${m.name} (${m.path})</option>`;
                });

                // Restore saved selection
                const savedYolo = localStorage.getItem('isitec_yolo_weights');
                if (savedYolo) yoloSelect.value = savedYolo;
                
                const savedDetr = localStorage.getItem('isitec_detr_weights');
                if (savedDetr) detrSelect.value = savedDetr;
            }
        } catch (e) {
            console.error("Failed to load models list", e);
        }
    }

    const btnSaveSettings = document.getElementById('btnSaveSettings');
    if (btnSaveSettings) {
        btnSaveSettings.addEventListener('click', () => {
            // Save Selects
            localStorage.setItem('isitec_yolo_weights', document.getElementById('set_yolo_weights').value);
            localStorage.setItem('isitec_detr_weights', document.getElementById('set_detr_weights').value);
            
            // Save Sliders
            ['yolo_imgsz', 'yolo_conf', 'detr_imgsz', 'detr_conf'].forEach(id => {
                localStorage.setItem(`isitec_${id}`, document.getElementById(`set_${id}`).value);
            });

            // Show confirmation
            const confirmMsg = document.getElementById('saveConfirm');
            confirmMsg.classList.remove('hidden');
            setTimeout(() => confirmMsg.classList.add('hidden'), 3000);
        });
    }

    loadSettings();

});
