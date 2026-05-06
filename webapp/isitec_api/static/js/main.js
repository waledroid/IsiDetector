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
        "start_tracking": "Start",
        "stop": "Stop",
        "analytics_overview": "Production Summary",
        "analytics_export": "Export Events",
        "period_today": "Today",
        "period_yesterday": "Yesterday",
        "rep_total": "Total",
        "rep_avg_fps": "Avg FPS",
        "rep_runtime": "Runtime",
        "rep_peak": "Peak hour",
        "rep_mix": "Mix",
        "rep_throughput": "Throughput",
        "rep_sessions": "Sessions",
        "rep_from": "From",
        "rep_to": "To",
        "rep_download_csv": "Download CSV",
        "sess_empty": "No sessions recorded yet.",
        "model_management": "Model Management",
        "model_soon": "Model Directory Viewer Coming Soon.",
        "system_settings": "System Settings",
        "mode_label": "Runtime mode",
        "mode_loaded_from": "Loaded:",
        "mode_banner_hint": "CPU vs GPU is auto-detected at container boot. Optimization defaults (skip-masks, threads, ByteTrack thresholds) live in <code>isidet/configs/inference/{common,cpu,gpu}.yaml</code> — not in this Settings panel.",
        "src_site_camera": "Site Camera",
        "site_camera_caption": "Saved camera (configurable in Settings → Camera)",
        "cam_settings": "Camera",
        "cam_settings_hint": "Tip: most IP cameras expose a sub-stream at a lower resolution — typically <code>stream=1</code> or <code>/102</code> in the URL path. On a CPU-only site PC, sub-stream gives much higher FPS.",
        "set_rtsp_url_label": "Default RTSP URL",
        "set_auto_start_label": "Auto-start stream on boot",
        "set_auto_start_hint": "Once enabled, click Start manually one time to record the model selection. After that, the stream comes up by itself every container boot — no operator click needed.",
        "set_roi_btn": "Set ROI",
        "roi_clear_btn": "Clear ROI",
        "set_roi_enabled_label": "Show \"Set ROI\" button on landing page",
        "set_clahe_enabled_label": "Apply CLAHE preprocess (glare / low-light correction)",
        "set_clahe_enabled_hint": "Boosts contrast in shadowed and reflective regions before the model sees the frame. Re-Start the stream after toggling.",
        "roi_current": "Current ROI:",
        "roi_none": "none (full frame)",
        "roi_drag_instruction": "Click and drag a rectangle over the conveyor belt area.",
        "roi_save": "Save ROI",
        "roi_cancel": "Cancel",
        "roi_need_stream": "Start the stream first to capture a snapshot.",
        "roi_saved": "ROI saved. Stop and Start the stream to apply.",
        "roi_cleared": "ROI cleared (full-frame mode).",
        "roi_cleared_alert": "ROI cleared. Stop and Start the stream to apply.",
        "roi_save_failed": "Could not save ROI: ",
        "roi_clear_failed": "Could not clear ROI: ",
        "sorter_settings": "Sorter (UDP target)",
        "sorter_settings_hint": "Each line crossing fires one ~60-byte JSON datagram <code>{class, id, ts}</code> to this address. Save → publisher retargets immediately, no stream restart needed. Test with <code>./net.sh test</code>.",
        "set_udp_host_label": "Sorter IP / hostname",
        "set_udp_port_label": "UDP port",
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
        "perf_empty": "Start a stream to see live metrics",
        "fs_prompt_text": "Fullscreen view?",
        "fs_prompt_accept": "Enter"
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
        "start_tracking": "Démarrer",
        "stop": "Arrêter",
        "analytics_overview": "Résumé de production",
        "analytics_export": "Exporter les événements",
        "period_today": "Aujourd'hui",
        "period_yesterday": "Hier",
        "rep_total": "Total",
        "rep_avg_fps": "IPS moy.",
        "rep_runtime": "Durée",
        "rep_peak": "Heure de pointe",
        "rep_mix": "Mélange",
        "rep_throughput": "Débit",
        "rep_sessions": "Sessions",
        "rep_from": "Du",
        "rep_to": "Au",
        "rep_download_csv": "Télécharger CSV",
        "sess_empty": "Aucune session enregistrée.",
        "model_management": "Gestion des modèles",
        "model_soon": "Visionneur de répertoire de modèles à venir.",
        "system_settings": "Paramètres du système",
        "mode_label": "Mode d'exécution",
        "mode_loaded_from": "Chargé :",
        "mode_banner_hint": "Le mode CPU/GPU est détecté automatiquement au démarrage du conteneur. Les paramètres d'optimisation (masques, threads, seuils ByteTrack) vivent dans <code>isidet/configs/inference/{common,cpu,gpu}.yaml</code> — pas dans ce panneau.",
        "src_site_camera": "Caméra du site",
        "site_camera_caption": "Caméra enregistrée (configurable dans Paramètres → Caméra)",
        "cam_settings": "Caméra",
        "cam_settings_hint": "Astuce : la plupart des caméras IP exposent un sous-flux à plus faible résolution — typiquement <code>stream=1</code> ou <code>/102</code> dans l'URL. Sur un PC site CPU-only, le sous-flux donne un FPS beaucoup plus élevé.",
        "set_rtsp_url_label": "URL RTSP par défaut",
        "set_auto_start_label": "Démarrage auto au boot",
        "set_auto_start_hint": "Une fois activé, cliquez Démarrer une fois manuellement pour mémoriser le modèle. Ensuite, le flux démarre tout seul à chaque redémarrage du conteneur — aucun clic opérateur nécessaire.",
        "set_roi_btn": "Définir ROI",
        "roi_clear_btn": "Effacer ROI",
        "set_roi_enabled_label": "Afficher le bouton « Définir ROI » sur la page d'accueil",
        "set_clahe_enabled_label": "Appliquer le prétraitement CLAHE (correction d'éblouissement / faible lumière)",
        "set_clahe_enabled_hint": "Renforce le contraste dans les zones d'ombre et de reflet avant que le modèle ne voie l'image. Redémarrez le flux après avoir basculé.",
        "roi_current": "ROI actuelle :",
        "roi_none": "aucune (image complète)",
        "roi_drag_instruction": "Cliquez et glissez pour tracer un rectangle sur la zone du convoyeur.",
        "roi_save": "Enregistrer ROI",
        "roi_cancel": "Annuler",
        "roi_need_stream": "Démarrez d'abord le flux pour capturer une image.",
        "roi_saved": "ROI enregistrée. Arrêter puis Démarrer le flux pour l'appliquer.",
        "roi_cleared": "ROI effacée (mode plein cadre).",
        "roi_cleared_alert": "ROI effacée. Arrêter puis Démarrer le flux pour l'appliquer.",
        "roi_save_failed": "Échec de l'enregistrement ROI : ",
        "roi_clear_failed": "Échec de l'effacement ROI : ",
        "sorter_settings": "Trieur (cible UDP)",
        "sorter_settings_hint": "Chaque franchissement de ligne envoie un datagramme JSON ~60 octets <code>{class, id, ts}</code> à cette adresse. Enregistrer → l'éditeur retargete immédiatement, pas de redémarrage du flux. Tester avec <code>./net.sh test</code>.",
        "set_udp_host_label": "IP / hôte du trieur",
        "set_udp_port_label": "Port UDP",
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
        "perf_empty": "Démarrez un flux pour voir les métriques en direct",
        "fs_prompt_text": "Affichage plein écran ?",
        "fs_prompt_accept": "Activer"
    }
};

const msgTrans = {
    en: {
        "msg_uploading": "Uploading file...",
        "msg_select_file": "Please select a file.",
        "msg_network_err": "Network error.",
        "msg_starting": "Starting inference stream...",
        "msg_enter_url": "Please enter a valid source URL/ID.",
        "placeholder_rtsp": "Enter RTSP URL (e.g. rtsp://192.168.1.108:554/stream)",
        "placeholder_cam": "Enter Camera ID (e.g. 0 or 1)"
    },
    fr: {
        "msg_uploading": "Téléchargement en cours...",
        "msg_select_file": "Veuillez sélectionner un fichier.",
        "msg_network_err": "Erreur réseau.",
        "msg_starting": "Démarrage automatique du flux d'inférence...",
        "msg_enter_url": "Veuillez entrer une URL / ID de source valide.",
        "placeholder_rtsp": "Entrez l'URL RTSP (ex. rtsp://192.168.1.108:554/stream)",
        "placeholder_cam": "Entrez l'ID de la caméra (ex. 0 ou 1)"
    }
};

document.addEventListener('DOMContentLoaded', () => {

    // --- I18N Translation Logic ---
    let currentLang = 'fr';
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
        if (currentSourceType === 'camera') {
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
    // French default translation runs at the end of this handler, after all
    // `let` declarations below are in scope — see setLanguage('fr') at the
    // bottom of DOMContentLoaded.

    // ── Dev-mode authentication ────────────────────────────────────────────
    const devModal  = document.getElementById('devModal');
    const devPassIn = document.getElementById('devPassword');
    const devError  = document.getElementById('devError');

    function devToken() { return sessionStorage.getItem('dev_token') || ''; }
    function devHeaders() { return devToken() ? { 'X-Dev-Token': devToken() } : {}; }

    function unlockDevUI() {
        document.querySelectorAll('.dev-only').forEach(el => el.style.display = '');
    }
    function lockDevUI() {
        sessionStorage.removeItem('dev_token');
        document.querySelectorAll('.dev-only').forEach(el => el.style.display = 'none');
        const activePanel = document.querySelector('.panel.active');
        if (activePanel && (activePanel.id === 'section-performance' || activePanel.id === 'section-settings')) {
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            document.querySelector('[data-target="section-live"]').classList.add('active');
            document.getElementById('section-live').classList.add('active');
        }
    }

    // Double-click logo to open modal
    document.querySelector('.header-logo').addEventListener('dblclick', () => {
        if (devToken()) { lockDevUI(); return; }
        devPassIn.value = '';
        devError.style.display = 'none';
        devModal.style.display = 'flex';
        devPassIn.focus();
    });

    // Enter → submit, Escape → cancel
    devPassIn.addEventListener('keydown', async (e) => {
        if (e.key === 'Escape') { devModal.style.display = 'none'; return; }
        if (e.key !== 'Enter') return;
        try {
            const res = await fetch('/api/dev-auth', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ password: devPassIn.value })
            });
            const d = await res.json();
            if (res.ok && d.token) {
                sessionStorage.setItem('dev_token', d.token);
                unlockDevUI();
                devModal.style.display = 'none';
            } else {
                devError.style.display = 'block';
            }
        } catch (e) {
            devError.style.display = 'block';
        }
    });

    // Close modal on overlay click
    devModal.addEventListener('click', (e) => { if (e.target === devModal) devModal.style.display = 'none'; });

    // On page load: re-validate existing token
    if (devToken()) {
        fetch('/api/dev-check', { headers: devHeaders() })
            .then(r => { if (r.ok) unlockDevUI(); else lockDevUI(); })
            .catch(() => lockDevUI());
    }

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
            if (targetId === 'section-analytics') fetchReport(currentReportPeriod);
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
    // Two chart modes — see isitec_app/static/js/main.js for full prose.
    const CHART_COLORS = {
        carton:  '#1b9bd8',
        polybag: '#ec407a',
    };
    const FALLBACK_COLOR = '#607d8b';
    const ctx = document.getElementById('analyticsChart').getContext('2d');

    let detectionChart = null;
    let currentChartPeriod = 'live';

    function _buildLiveChart() {
        if (detectionChart) detectionChart.destroy();
        detectionChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Count',
                    data: [],
                    backgroundColor: [CHART_COLORS.carton, CHART_COLORS.polybag, '#43a047'],
                    borderColor:     [CHART_COLORS.carton, CHART_COLORS.polybag, '#43a047'],
                    borderWidth: 1,
                    borderRadius: 4,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
                plugins: { legend: { display: false } },
            },
        });
    }

    function _buildTimeseriesChart(payload) {
        if (detectionChart) detectionChart.destroy();
        const classes = Object.keys(payload.series);
        const datasets = classes.map(c => ({
            label: c.charAt(0).toUpperCase() + c.slice(1),
            data: payload.series[c],
            backgroundColor: CHART_COLORS[c] || FALLBACK_COLOR,
            borderRadius: 2,
        }));
        detectionChart = new Chart(ctx, {
            type: 'bar',
            data: { labels: payload.buckets, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { stacked: true, ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 12 } },
                    y: { stacked: true, beginAtZero: true, ticks: { precision: 0 } },
                },
                plugins: {
                    legend: { display: true, position: 'top' },
                    tooltip: { mode: 'index', intersect: false },
                },
            },
        });
    }

    _buildLiveChart();

    // Chart Filter Buttons Logic
    // Live mode updates via the 500 ms /api/stats poll. Historical modes
    // (24h / 7d / 30d) auto-refresh every 5 s so the rightmost bucket
    // visibly fills as events arrive.
    let chartHistoricalInterval = null;
    function _stopHistoricalChartPolling() {
        if (chartHistoricalInterval) {
            clearInterval(chartHistoricalInterval);
            chartHistoricalInterval = null;
        }
    }
    function _startHistoricalChartPolling(period) {
        _stopHistoricalChartPolling();
        chartHistoricalInterval = setInterval(() => updateChartData(period), 5000);
    }

    const filterBtns = document.querySelectorAll('.filter-btn');
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentChartPeriod = btn.getAttribute('data-period');
            updateChartData(currentChartPeriod);
            if (currentChartPeriod === 'live') {
                _stopHistoricalChartPolling();
            } else {
                _startHistoricalChartPolling(currentChartPeriod);
            }
        });
    });

    async function updateChartData(period) {
        try {
            const res = await fetch(`/api/chart?period=${period}`);
            const data = await res.json();
            if (data.status !== 'success') return;

            if (data.view === 'timeseries') {
                _buildTimeseriesChart(data);
            } else {
                _buildLiveChart();
                const counts = data.data || {};
                detectionChart.data.labels = Object.keys(counts).map(l => l.toUpperCase());
                detectionChart.data.datasets[0].data = Object.values(counts);
                detectionChart.update('none');
            }
        } catch (e) {
            console.error('Error fetching chart data', e);
        }
    }

    // ── WebSocket connections ────────────────────────────────────────────────
    let videoWs = null;
    let statsWs = null;

    const videoCanvas = document.getElementById('videoCanvas');
    const vCtx = videoCanvas.getContext('2d');

    // ROI capture pauses the WebSocket-to-canvas draw so the snapshot stays
    // visible while the operator clicks the 4 corners.
    let roiCaptureActive = false;
    const statCartons = document.getElementById('statCartons');
    const statPolybags = document.getElementById('statPolybags');
    const statLast = document.getElementById('statLast');

    function wsProto() {
        return location.protocol === 'https:' ? 'wss:' : 'ws:';
    }

    // ── Video WebSocket ──────────────────────────────────────────────────────
    function connectVideoWs() {
        if (videoWs && videoWs.readyState <= WebSocket.OPEN) return;

        videoWs = new WebSocket(`${wsProto()}//${location.host}/ws/video`);
        videoWs.binaryType = 'blob';

        videoWs.onmessage = (event) => {
            // Skip frame draws while the operator is configuring an ROI —
            // the snapshot displayed under the click markers must not be
            // overwritten by incoming live frames.
            if (roiCaptureActive) return;
            const blob = event.data;
            const url = URL.createObjectURL(blob);
            const img = new Image();
            img.onload = () => {
                if (videoCanvas.width !== img.width || videoCanvas.height !== img.height) {
                    videoCanvas.width = img.width;
                    videoCanvas.height = img.height;
                }
                vCtx.drawImage(img, 0, 0);
                URL.revokeObjectURL(url);
            };
            img.src = url;
        };

        videoWs.onclose = () => {
            videoWs = null;
            // Auto-reconnect after 2s if stream is presumably still running
            setTimeout(() => {
                if (!videoWs) connectVideoWs();
            }, 2000);
        };

        videoWs.onerror = () => {
            videoWs.close();
        };
    }

    // ── Stats WebSocket ──────────────────────────────────────────────────────
    function connectStatsWs() {
        if (statsWs && statsWs.readyState <= WebSocket.OPEN) return;

        statsWs = new WebSocket(`${wsProto()}//${location.host}/ws/stats`);

        statsWs.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateStatsUI(data);
        };

        statsWs.onclose = () => {
            statsWs = null;
            setTimeout(() => {
                if (!statsWs) connectStatsWs();
            }, 3000);
        };

        statsWs.onerror = () => {
            statsWs.close();
        };
    }

    function updateStatsUI(data) {
        if (data.counts) {
            let cartonCount = 0;
            let polybagCount = 0;
            for (let key in data.counts) {
                if (key.toLowerCase().includes('carton')) cartonCount += data.counts[key];
                if (key.toLowerCase().includes('polybag') || key.toLowerCase().includes('bag')) polybagCount += data.counts[key];
            }
            statCartons.textContent = cartonCount;
            statPolybags.textContent = polybagCount;

            if (currentChartPeriod === 'live') {
                const labels = Object.keys(data.counts);
                const values = Object.values(data.counts);
                detectionChart.data.labels = labels.map(l => l.toUpperCase());
                detectionChart.data.datasets[0].data = values;
                detectionChart.update('none');
            }
        }

        const udpEl = document.getElementById('udpStatus');
        if (data.last_detected) {
            const cls = data.last_detected.class.toUpperCase();
            const ts = data.last_detected.time.split('T')[1].split('.')[0];
            const id = data.last_detected.id;
            const idStr = (id !== undefined && id !== null) ? ` #${id}` : '';
            statLast.textContent = `${cls}${idStr} - ${data.last_detected.time}`;
            statLast.removeAttribute('data-i18n');
            udpEl.textContent = `UDP \u2192 ${cls}${idStr} @ ${ts}`;
            udpEl.style.color = '#43a047';
        } else if (udpEl) {
            udpEl.textContent = 'UDP: idle';
            udpEl.style.color = '#9aa0a6';
        }
    }

    function closeWebSockets() {
        if (videoWs) { videoWs.onclose = null; videoWs.close(); videoWs = null; }
        if (statsWs) { statsWs.onclose = null; statsWs.close(); statsWs = null; }
    }

    // Load initial empty chart
    updateChartData('live');

    // On page load, check if an inference session is already running and restore
    async function restoreSession() {
        try {
            const res = await fetch('/api/stats');
            const data = await res.json();

            if (data.is_running) {
                connectVideoWs();
                connectStatsWs();

                if (data.counts) {
                    let cartonCount = 0, polybagCount = 0;
                    for (let key in data.counts) {
                        if (key.toLowerCase().includes('carton')) cartonCount += data.counts[key];
                        if (key.toLowerCase().includes('polybag') || key.toLowerCase().includes('bag')) polybagCount += data.counts[key];
                    }
                    statCartons.textContent = cartonCount;
                    statPolybags.textContent = polybagCount;

                    detectionChart.data.labels = Object.keys(data.counts).map(l => l.toUpperCase());
                    detectionChart.data.datasets[0].data = Object.values(data.counts);
                    detectionChart.update('none');
                }

                if (data.last_detected) {
                    const id = data.last_detected.id;
                    const idStr = (id !== undefined && id !== null) ? ` #${id}` : '';
                    statLast.textContent = `${data.last_detected.class.toUpperCase()}${idStr} - ${data.last_detected.time}`;
                    statLast.removeAttribute('data-i18n');
                }

                startPerfPolling();
                fetchPerformance();
            }
        } catch (e) {
            console.error("Session restore check failed:", e);
        }
    }
    restoreSession();

    // --- Fullscreen onboarding prompt (non-blocking) ---
    // Browser requires a user gesture for requestFullscreen(), so we can't
    // auto-enter on load — we surface a small corner card and the operator
    // clicks "Enter" or the dismiss "×".
    (function initFullscreenPrompt() {
        const prompt = document.getElementById('fs-prompt');
        if (!prompt) return;
        if (document.fullscreenElement || document.webkitFullscreenElement) return;
        const el = document.documentElement;
        const requestFS = el.requestFullscreen
                       || el.webkitRequestFullscreen
                       || el.mozRequestFullScreen
                       || el.msRequestFullscreen;
        if (!requestFS) { console.warn('[fs] Fullscreen API unavailable in this browser'); return; }
        prompt.classList.remove('hidden');
        prompt.querySelector('.fs-prompt-accept').addEventListener('click', () => {
            try {
                const result = requestFS.call(el);
                // Modern browsers return a Promise; older WebKit returns undefined.
                if (result && typeof result.catch === 'function') {
                    result.catch(err => console.warn('[fs] requestFullscreen rejected:', err));
                }
            } catch (err) {
                console.warn('[fs] requestFullscreen threw:', err);
            }
            prompt.classList.add('hidden');
        });
        prompt.querySelector('.fs-prompt-dismiss').addEventListener('click', () => {
            prompt.classList.add('hidden');
        });
    })();

    // --- Source Selection Logic ---
    const sourceBtns = document.querySelectorAll('.source-btn');
    const fileDropArea = document.getElementById('fileDropArea');
    const sourceFile = document.getElementById('sourceFile');
    const sourceText = document.getElementById('sourceText');
    const siteCamCaption = document.getElementById('siteCameraCaption');
    const fileMsg = document.querySelector('.file-msg');

    // Default source = the saved Site Camera (RTSP URL from settings.json).
    // 'rtsp' type removed — its text input duplicated the saved URL. Other
    // 3 buttons are ad-hoc overrides (Image / Video / USB Camera).
    let currentSourceType = 'site_camera';
    let uploadedFilePath = '';

    function _showSourceUI(type) {
        if (fileDropArea) fileDropArea.style.display = 'none';
        if (sourceText) sourceText.style.display = 'none';
        if (siteCamCaption) siteCamCaption.style.display = 'none';

        if (type === 'image') {
            fileDropArea.style.display = 'flex';
            sourceFile.setAttribute('accept', 'image/*');
            fileMsg.textContent = translations[currentLang].drag_drop;
        } else if (type === 'video') {
            fileDropArea.style.display = 'flex';
            sourceFile.setAttribute('accept', 'video/mp4,video/x-m4v,video/*');
            fileMsg.textContent = translations[currentLang].drag_drop;
        } else if (type === 'camera') {
            sourceText.style.display = 'block';
            sourceText.placeholder = msgTrans[currentLang].placeholder_cam;
            sourceText.value = '0';
        } else {
            // site_camera (default) — no input field; backend uses settings.rtsp_url
            if (siteCamCaption) siteCamCaption.style.display = 'flex';
        }
    }

    _showSourceUI(currentSourceType);

    sourceBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            sourceBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentSourceType = btn.getAttribute('data-type');
            _showSourceUI(currentSourceType);
        });
    });

    sourceFile.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            fileMsg.textContent = e.target.files[0].name;
            uploadedFilePath = '';
        }
    });

    // --- API Commands ---
    const btnStart = document.getElementById('btnStart');
    const btnStop = document.getElementById('btnStop');

    btnStart.addEventListener('click', async () => {
        const model_type = document.getElementById('modelSelect').value;

        let weights = '';
        let imgsz = null;
        let conf = null;

        if (model_type === 'yolo') {
            weights = localStorage.getItem('isitec_yolo_weights') || '';
            // imgsz is now mode-driven (cpu.yaml / gpu.yaml) — backend resolves it
            conf = localStorage.getItem('isitec_yolo_conf');
        } else if (model_type === 'Detr') {
            weights = localStorage.getItem('isitec_detr_weights') || '';
            // imgsz is now mode-driven — backend resolves it
            conf = localStorage.getItem('isitec_detr_conf');
        }

        let finalSource = '';

        if (currentSourceType === 'site_camera') {
            // Empty source → backend resolves from settings.rtsp_url.
            finalSource = '';
        } else if (currentSourceType === 'image' || currentSourceType === 'video') {
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
            // 'camera' (USB) — sourceText holds the device index
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

                // Connect WebSockets for video + stats
                connectVideoWs();
                connectStatsWs();

                statCartons.textContent = "0";
                statPolybags.textContent = "0";
                statLast.textContent = translations[currentLang].stat_none;
                startPerfPolling();
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
                closeWebSockets();
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
        const perfVisible = document.getElementById('section-performance')?.classList.contains('active');
        const analyticsVisible = document.getElementById('section-analytics')?.classList.contains('active');
        if (!perfVisible && !analyticsVisible) return;
        try {
            const res = await fetch('/api/performance', { headers: devHeaders() });
            if (res.status === 403) return;
            const d = await res.json();

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

    function pmRow(label, valueHtml) {
        return `<div class="pm-row"><span class="pm-label">${label}</span><span class="pm-value">${valueHtml}</span></div>`;
    }
    function fmt(val, unit, dec) {
        if (val == null) return '<span class="pm-na">\u2014</span>';
        return Number(val).toFixed(dec !== undefined ? dec : 1) + (unit || '');
    }
    function fmtPct(val) { return fmt(val, '%', 0); }

    function buildSessionRows(d) {
        return pmRow('Uptime',       d.uptime_fmt || '<span class="pm-na">\u2014</span>')
             + pmRow('Status',       d.is_running ? '\ud83d\udfe2 LIVE' : '\u23f9 Idle')
             + pmRow('Errors',       fmt(d.error_count, '', 0))
             + pmRow('CUDA OOM',     fmt(d.cuda_oom_count, '', 0))
             + pmRow('Heartbeat',    d.heartbeat_age_s != null ? fmt(d.heartbeat_age_s, 's', 0) : '<span class="pm-na">\u2014</span>');
    }

    function deltaTag(val, unit, invert) {
        if (val == null) return '';
        const sign = val >= 0 ? '+' : '';
        const cls  = val > 0 ? 'delta-up' : val < 0 ? 'delta-down' : '';
        return `<span class="pm-delta ${cls}">${sign}${Math.round(val)}${unit}</span>`;
    }

    function buildProgressBar(label, pct, valueHtml, thresholds) {
        const [warnAt, critAt] = thresholds || [80, 90];
        const barClass = pct > critAt ? 'bar-red' : pct > warnAt ? 'bar-yellow' : 'bar-green';
        return `<div class="pm-row" style="flex-direction:column; gap:4px;">
            <div style="display:flex; justify-content:space-between; width:100%;">
                <span class="pm-label">${label}</span>
                <span class="pm-value">${valueHtml}</span>
            </div>
            <div class="pm-bar-wrap">
                <div class="pm-bar-track"><div class="pm-bar-fill ${barClass}" style="width:${pct}%"></div></div>
                <span class="pm-bar-label">${pct}%</span>
            </div>
        </div>`;
    }

    function buildHardwareRows(d) {
        let html = '';

        if (d.has_gpu) {
            if (d.vram_used_mb != null && d.vram_total_mb != null) {
                const pct = Math.round(d.vram_pct || 0);
                const delta = d.vram_delta_mb != null ? deltaTag(d.vram_delta_mb / 1024, ' GB', false) : '';
                html += buildProgressBar('VRAM',
                    pct, `${(d.vram_used_mb/1024).toFixed(1)} / ${(d.vram_total_mb/1024).toFixed(1)} GB${delta}`,
                    [70, 85]);
            }
            if (d.gpu_util_pct != null) {
                const pct = Math.round(d.gpu_util_pct);
                const delta = deltaTag(d.gpu_util_delta_pct, '%', false);
                html += buildProgressBar('GPU Util', pct, `${pct}%${delta}`, [80, 90]);
            }
            const tempVal = d.gpu_temp_c != null
                ? fmt(d.gpu_temp_c, '\u00b0C', 0) + deltaTag(d.temp_delta_c, '\u00b0C', false)
                : '<span class="pm-na">\u2014</span>';
            html += pmRow('GPU Temp', tempVal);
        } else {
            if (d.cpu_pct != null) {
                const pct = Math.round(d.cpu_pct);
                html += buildProgressBar('CPU Util', pct, `${pct}%`, [80, 95]);
            }
            if (d.cpu_model) {
                const model = String(d.cpu_model).replace(/\(R\)|\(TM\)/g, '').replace(/\s+/g, ' ').trim();
                html += pmRow('CPU Model', `<span style="font-size: 12px;">${model}</span>`);
            }
            if (d.cpu_freq_mhz != null) {
                html += pmRow('CPU Freq', fmt(d.cpu_freq_mhz, ' MHz', 0));
            }
            if (d.cpu_cores != null) {
                html += pmRow('CPU Cores', fmt(d.cpu_cores, '', 0));
            }
            html += pmRow('CPU Temp', d.cpu_temp_c != null ? fmt(d.cpu_temp_c, '\u00b0C', 0) : '<span class="pm-na">\u2014</span>');
            // ML feature flags \u2014 answers "is INT8 quantization worth it on this box?"
            // VNNI / AMX = green (INT8 2-3\u00d7 wins). AVX-512 = amber. AVX2-only = grey.
            if (Array.isArray(d.cpu_flags) && d.cpu_flags.length) {
                const flags = d.cpu_flags;
                const tag = (f, color) => `<span style="display:inline-block; padding:1px 6px; margin-right:4px; border-radius:3px; background:${color}; color:#fff; font-size:11px; font-family:monospace;">${f}</span>`;
                const colored = flags.map(f => {
                    if (f === 'avx512_vnni' || f.startsWith('amx_')) return tag(f, '#1b8a3a');
                    if (f.startsWith('avx512')) return tag(f, '#c4831f');
                    return tag(f, '#5c6370');
                }).join('');
                html += pmRow('ML Features', `<div style="line-height:1.8;">${colored}</div>`);
            }
        }

        if (d.ram_used_mb != null && d.ram_total_mb != null) {
            const pct = Math.round(d.ram_pct || 0);
            const delta = d.ram_delta_mb != null ? deltaTag(d.ram_delta_mb / 1024, ' GB', false) : '';
            html += buildProgressBar('System RAM', pct,
                `${(d.ram_used_mb/1024).toFixed(1)} / ${(d.ram_total_mb/1024).toFixed(1)} GB${delta}`,
                [85, 95]);
        } else {
            html += pmRow('System RAM', '<span class="pm-na">\u2014</span>');
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
             + pmRow('Low-Conf Rate',  d.low_conf_rate != null ? (d.low_conf_rate * 100).toFixed(1) + '%' : '<span class="pm-na">\u2014</span>')
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
        if (keys.length === 0) return '<div class="pm-row"><span class="pm-label">No data</span><span class="pm-value pm-na">\u2014</span></div>';
        return keys.map(k => {
            const rateStr = rates[k] != null ? ` (${rates[k]}/hr)` : '';
            return pmRow(k.charAt(0).toUpperCase() + k.slice(1), `${totals[k]}${rateStr}`);
        }).join('');
    }

    function updateSessionsTable(sessions) {
        const tbody = document.getElementById('sessionsBody');
        if (!tbody) return;
        const emptyMsg = (translations[currentLang] && translations[currentLang].sess_empty)
            || 'No sessions recorded yet.';
        if (!sessions || sessions.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color: var(--text-light); padding: 24px;">${emptyMsg}</td></tr>`;
            return;
        }
        tbody.innerHTML = sessions.map(s => {
            const modelClass = (s.model || '').toLowerCase().includes('rfdetr') ? 'sess-model-rfdetr' : 'sess-model-yolo';
            const countsStr  = s.counts ? Object.entries(s.counts).map(([k,v]) => `${k}: ${v}`).join(', ') : '\u2014';
            return `<tr>
                <td>${s.date || '\u2014'}</td>
                <td class="${modelClass}">${(s.model || '\u2014').toUpperCase()}</td>
                <td>${s.duration_h != null ? s.duration_h.toFixed(1) + 'h' : '\u2014'}</td>
                <td>${s.fps != null ? s.fps.toFixed(1) : '\u2014'}</td>
                <td>${s.avg_confidence != null ? s.avg_confidence.toFixed(3) : '\u2014'}</td>
                <td>${s.id_ratio != null ? s.id_ratio.toFixed(2) : '\u2014'}</td>
                <td>${countsStr}</td>
            </tr>`;
        }).join('');
    }

    // \u2500\u2500 Production Summary / Report \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    let currentReportPeriod = 'today';

    async function fetchReport(period) {
        try {
            const res = await fetch(`/api/report?period=${encodeURIComponent(period)}`);
            const d = await res.json();
            if (d.status !== 'success') return;

            document.getElementById('repCarton').textContent    = (d.counts.carton || 0).toLocaleString();
            document.getElementById('repPolybag').textContent   = (d.counts.polybag || 0).toLocaleString();
            document.getElementById('repTotal').textContent     = (d.counts.total  || 0).toLocaleString();
            document.getElementById('repFps').textContent       = d.avg_fps != null ? d.avg_fps.toFixed(1) : '\u2014';
            document.getElementById('repRuntime').textContent   = (d.total_runtime_h || 0).toFixed(1) + 'h';

            const peakTxt = d.peak && d.peak.hour ? `${d.peak.hour} (${d.peak.events})` : '\u2014';
            document.getElementById('repPeak').textContent       = peakTxt;
            const carton = d.mix_pct?.carton ?? 0;
            const polybag = d.mix_pct?.polybag ?? 0;
            document.getElementById('repMix').textContent        = `${carton}% / ${polybag}%`;
            document.getElementById('repThroughput').textContent = `${d.throughput_per_hour || 0}/h`;
            document.getElementById('repSessionsCount').textContent = d.sessions_count || 0;

            updateSessionsTable(d.recent_sessions || []);

            const fromEl = document.getElementById('exportFrom');
            const toEl   = document.getElementById('exportTo');
            if (fromEl && !fromEl.value) fromEl.value = d.window.from.slice(0, 10);
            if (toEl && !toEl.value) {
                const toDate = new Date(d.window.to);
                toDate.setDate(toDate.getDate() - 1);
                toEl.value = toDate.toISOString().slice(0, 10);
            }
        } catch (e) {
            console.error('Report fetch failed', e);
        }
    }

    document.querySelectorAll('#reportPeriodBtns .filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('#reportPeriodBtns .filter-btn')
                .forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentReportPeriod = btn.getAttribute('data-period');
            document.getElementById('exportFrom').value = '';
            document.getElementById('exportTo').value = '';
            fetchReport(currentReportPeriod);
        });
    });

    const btnExportCsv = document.getElementById('btnExportCsv');
    if (btnExportCsv) {
        btnExportCsv.addEventListener('click', () => {
            const f = document.getElementById('exportFrom').value;
            const t = document.getElementById('exportTo').value;
            if (!f || !t) { showMessage('Please pick a From and To date.', 'warning'); return; }
            window.location.href = `/api/events/export?from=${encodeURIComponent(f)}&to=${encodeURIComponent(t)}`;
        });
    }


    // Always start perf polling on load
    startPerfPolling();

    // ── Settings Panel Logic ────────────────────────────────────────────────

    // ── Tracking Line Controls ──────────────────────────────────────────
    const lineSlider = document.getElementById('linePositionSlider');
    const linePosVal = document.getElementById('linePositionVal');
    const beltDirectionGroup = document.getElementById('beltDirectionGroup');
    let lineOrientation = 'vertical';
    let beltDirection = 'left_to_right';

    const BELT_DIRECTION_OPTIONS = {
        vertical: [
            ['left_to_right', 'Left → Right'],
            ['right_to_left', 'Right → Left'],
        ],
        horizontal: [
            ['top_to_bottom', 'Top → Bottom'],
            ['bottom_to_top', 'Bottom → Top'],
        ],
    };

    function renderBeltDirectionButtons() {
        const options = BELT_DIRECTION_OPTIONS[lineOrientation];
        const valid = options.some(([v]) => v === beltDirection);
        if (!valid) beltDirection = options[0][0];

        beltDirectionGroup.innerHTML = '';
        options.forEach(([value, label]) => {
            const btn = document.createElement('button');
            btn.className = 'line-btn' + (value === beltDirection ? ' active' : '');
            btn.textContent = label;
            btn.addEventListener('click', () => {
                beltDirection = value;
                renderBeltDirectionButtons();
                updateLinePreview(lineOrientation, parseInt(lineSlider.value), beltDirection);
            });
            beltDirectionGroup.appendChild(btn);
        });
    }

    function updateLinePreview(orientation, position, belt) {
        fetch('/api/line', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', ...devHeaders() },
            body: JSON.stringify({
                orientation,
                position: position / 100,
                belt_direction: belt,
            })
        }).catch(e => console.error('Line update failed', e));
    }

    document.getElementById('btnLineVertical').addEventListener('click', () => {
        lineOrientation = 'vertical';
        document.getElementById('btnLineVertical').classList.add('active');
        document.getElementById('btnLineHorizontal').classList.remove('active');
        renderBeltDirectionButtons();
        updateLinePreview(lineOrientation, parseInt(lineSlider.value), beltDirection);
    });

    document.getElementById('btnLineHorizontal').addEventListener('click', () => {
        lineOrientation = 'horizontal';
        document.getElementById('btnLineHorizontal').classList.add('active');
        document.getElementById('btnLineVertical').classList.remove('active');
        renderBeltDirectionButtons();
        updateLinePreview(lineOrientation, parseInt(lineSlider.value), beltDirection);
    });

    document.getElementById('btnLineBack').addEventListener('click', () => {
        lineSlider.value = Math.max(10, parseInt(lineSlider.value) - 5);
        linePosVal.textContent = lineSlider.value;
        updateLinePreview(lineOrientation, parseInt(lineSlider.value), beltDirection);
    });

    document.getElementById('btnLineForward').addEventListener('click', () => {
        lineSlider.value = Math.min(90, parseInt(lineSlider.value) + 5);
        linePosVal.textContent = lineSlider.value;
        updateLinePreview(lineOrientation, parseInt(lineSlider.value), beltDirection);
    });

    lineSlider.addEventListener('input', (e) => {
        linePosVal.textContent = e.target.value;
    });
    lineSlider.addEventListener('change', (e) => {
        updateLinePreview(lineOrientation, parseInt(e.target.value), beltDirection);
    });

    renderBeltDirectionButtons();

    ['yolo_imgsz', 'yolo_conf', 'detr_imgsz', 'detr_conf'].forEach(id => {
        const slider = document.getElementById(`set_${id}`);
        const valSpan = document.getElementById(`val_${id}`);
        if (slider && valSpan) {
            slider.addEventListener('input', (e) => {
                valSpan.textContent = id.includes('imgsz') ? `${e.target.value}px` : parseFloat(e.target.value).toFixed(2);
            });
        }
    });
    // CPU threads slider — plain integer display (no `px` / `.XX` suffix)
    function applySliderValue(id, val) {
        const slider = document.getElementById(`set_${id}`);
        const valSpan = document.getElementById(`val_${id}`);
        if (slider && valSpan && val != null) {
            slider.value = val;
            valSpan.textContent = id.includes('imgsz') ? `${val}px` : parseFloat(val).toFixed(2);
        }
    }

    // Mode banner — fetch /api/mode and populate the read-only banner at the top
    // of the Settings panel. Also hide all .gpu-only elements when the resolved
    // mode is 'cpu' (RF-DETR / Mode 2 entry / RF-DETR config group).
    async function applyMode() {
        try {
            const res = await fetch('/api/mode');
            if (!res.ok) return;
            const data = await res.json();
            const mode = (data.mode || 'unknown').toLowerCase();
            const valEl = document.getElementById('mode_banner_value');
            const viaEl = document.getElementById('mode_banner_via');
            const filesEl = document.getElementById('mode_banner_files');
            if (valEl) valEl.textContent = mode === 'cpu' ? '⚡ CPU mode' : (mode === 'gpu' ? '🟢 GPU mode' : `❓ ${mode}`);
            if (viaEl) viaEl.textContent = data.detected_via ? `(detected via ${data.detected_via})` : '';
            if (filesEl) filesEl.textContent = (data.config_files || []).join(' + ') || '—';

            const gpuOnlyEls = document.querySelectorAll('.gpu-only');
            gpuOnlyEls.forEach(el => {
                el.style.display = (mode === 'cpu') ? 'none' : '';
            });
        } catch (e) {
            console.warn('Could not load /api/mode', e);
        }
    }

    async function loadSettings() {
        await applyMode();

        let serverSettings = {};
        try {
            const sRes = await fetch('/api/settings');
            const sData = await sRes.json();
            if (sData.status === 'success') serverSettings = sData.settings || {};
        } catch (e) { console.warn('Could not load server settings', e); }

        // Apply confidence sliders. Imgsz/cpu_threads/skip_* are mode-driven now.
        ['yolo_conf', 'detr_conf'].forEach(id => {
            const val = serverSettings[id] || localStorage.getItem(`isitec_${id}`);
            if (val != null) applySliderValue(id, val);
        });
        const rtspUrlEl = document.getElementById('set_rtsp_url');
        if (rtspUrlEl && serverSettings.rtsp_url) rtspUrlEl.value = serverSettings.rtsp_url;
        const autoStartEl = document.getElementById('set_auto_start');
        if (autoStartEl) autoStartEl.checked = !!serverSettings.auto_start;
        // CLAHE: glare/low-light preprocess. Stop+Start stream to apply.
        const claheEl = document.getElementById('set_clahe_enabled');
        if (claheEl) claheEl.checked = !!serverSettings.clahe_enabled;
        // ROI: toggle reveals the Live-page "Set ROI" + "Clear ROI" buttons + show current bbox.
        const roiEnabledEl = document.getElementById('set_roi_enabled');
        const setRoiBtn = document.getElementById('btnSetROI');
        const clearRoiBtn = document.getElementById('btnClearROI');
        const roiCurrentEl = document.getElementById('disp_roi_current');
        if (roiEnabledEl) roiEnabledEl.checked = !!serverSettings.roi_enabled;
        if (setRoiBtn) setRoiBtn.style.display = serverSettings.roi_enabled ? 'flex' : 'none';
        if (clearRoiBtn) clearRoiBtn.style.display = serverSettings.roi_enabled ? 'flex' : 'none';
        if (roiCurrentEl) {
            const pts = serverSettings.roi_points;
            if (Array.isArray(pts) && pts.length === 4) {
                const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
                const x1 = Math.min(...xs), x2 = Math.max(...xs);
                const y1 = Math.min(...ys), y2 = Math.max(...ys);
                roiCurrentEl.textContent = `x=[${x1},${x2}] y=[${y1},${y2}] (${x2-x1}×${y2-y1})`;
                roiCurrentEl.removeAttribute('data-i18n');
            } else {
                roiCurrentEl.setAttribute('data-i18n', 'roi_none');
                roiCurrentEl.textContent = (translations[currentLang] && translations[currentLang]['roi_none']) || 'none (full frame)';
            }
        }
        const udpHostEl = document.getElementById('set_udp_host');
        if (udpHostEl) udpHostEl.value = serverSettings.udp_host ?? '127.0.0.1';
        const udpPortEl = document.getElementById('set_udp_port');
        if (udpPortEl) udpPortEl.value = serverSettings.udp_port ?? 9502;

        // Restore line settings
        const savedOrientation = serverSettings.line_orientation || 'vertical';
        const savedPosition = Math.round((serverSettings.line_position || 0.5) * 100);
        const savedBeltDir = serverSettings.belt_direction
            || (savedOrientation === 'vertical' ? 'left_to_right' : 'top_to_bottom');
        lineSlider.value = savedPosition;
        linePosVal.textContent = savedPosition;
        lineOrientation = savedOrientation;
        beltDirection = savedBeltDir;
        document.getElementById('btnLineVertical').classList.toggle('active', savedOrientation === 'vertical');
        document.getElementById('btnLineHorizontal').classList.toggle('active', savedOrientation === 'horizontal');
        renderBeltDirectionButtons();

        try {
            const res = await fetch('/api/models');
            const data = await res.json();
            if (data.status === 'success') {
                const yoloSelect = document.getElementById('set_yolo_weights');
                const detrSelect = document.getElementById('set_detr_weights');

                // Group models by file extension so operators can see at
                // a glance which format they're picking — different formats
                // trade off size, speed, and hardware compatibility.
                const FORMAT_GROUPS = [
                    { ext: '.engine', label: '⚡ TensorRT (.engine) — NVIDIA GPU, fastest' },
                    { ext: '.pt',     label: '🧠 PyTorch (.pt) — native YOLO, flexible' },
                    { ext: '.pth',    label: '🧠 PyTorch (.pth) — native RF-DETR' },
                    { ext: '.xml',    label: '🟦 OpenVINO (.xml) — Intel CPU, fastest' },
                    { ext: '.onnx',   label: '🔷 ONNX (.onnx) — portable GPU/CPU' },
                ];
                function populateSelect(selectEl, models) {
                    selectEl.innerHTML = '<option value="">Auto-detect</option>';
                    FORMAT_GROUPS.forEach(group => {
                        const matching = models.filter(m => m.name.toLowerCase().endsWith(group.ext));
                        if (matching.length === 0) return;
                        const og = document.createElement('optgroup');
                        og.label = `${group.label}  (${matching.length})`;
                        matching.forEach(m => {
                            const opt = document.createElement('option');
                            opt.value = m.path;
                            opt.textContent = `${m.name}  —  ${m.path}`;
                            og.appendChild(opt);
                        });
                        selectEl.appendChild(og);
                    });
                    const other = models.filter(m =>
                        !FORMAT_GROUPS.some(g => m.name.toLowerCase().endsWith(g.ext))
                    );
                    if (other.length) {
                        const og = document.createElement('optgroup');
                        og.label = `❓ Other  (${other.length})`;
                        other.forEach(m => {
                            const opt = document.createElement('option');
                            opt.value = m.path;
                            opt.textContent = `${m.name}  —  ${m.path}`;
                            og.appendChild(opt);
                        });
                        selectEl.appendChild(og);
                    }
                }
                populateSelect(yoloSelect, data.yolo_models);
                populateSelect(detrSelect, data.rfdetr_models);

                const savedYolo = serverSettings.yolo_weights || localStorage.getItem('isitec_yolo_weights');
                if (savedYolo) yoloSelect.value = savedYolo;

                const savedDetr = serverSettings.rfdetr_weights || localStorage.getItem('isitec_detr_weights');
                if (savedDetr) detrSelect.value = savedDetr;
            }
        } catch (e) {
            console.error("Failed to load models list", e);
        }
    }

    const btnSaveSettings = document.getElementById('btnSaveSettings');
    if (btnSaveSettings) {
        btnSaveSettings.addEventListener('click', async () => {
            // imgsz / cpu_threads / skip_masks / skip_traces are mode-driven.
            const detrWeightsEl = document.getElementById('set_detr_weights');
            const detrConfEl = document.getElementById('set_detr_conf');
            const settings = {
                yolo_weights:  document.getElementById('set_yolo_weights').value,
                rfdetr_weights: detrWeightsEl ? detrWeightsEl.value : '',
                yolo_conf:     parseFloat(document.getElementById('set_yolo_conf').value),
                detr_conf:     detrConfEl ? parseFloat(detrConfEl.value) : 0.35,
                line_orientation: lineOrientation,
                line_position: parseInt(lineSlider.value) / 100,
                belt_direction: beltDirection,
                rtsp_url:      document.getElementById('set_rtsp_url').value.trim(),
                auto_start:    document.getElementById('set_auto_start').checked,
                roi_enabled:   document.getElementById('set_roi_enabled').checked,
                clahe_enabled: document.getElementById('set_clahe_enabled').checked,
                udp_host:      document.getElementById('set_udp_host').value.trim(),
                udp_port:      parseInt(document.getElementById('set_udp_port').value),
            };

            localStorage.setItem('isitec_yolo_weights', settings.yolo_weights);
            localStorage.setItem('isitec_detr_weights', settings.rfdetr_weights || '');
            ['yolo_conf', 'detr_conf'].forEach(id => {
                if (settings[id] != null) localStorage.setItem(`isitec_${id}`, settings[id]);
            });

            try {
                await fetch('/api/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...devHeaders() },
                    body: JSON.stringify(settings)
                });
            } catch (e) { console.warn('Server settings save failed', e); }

            // Live-reflect the ROI toggle on the landing page without a reload.
            const setRoiBtnNow = document.getElementById('btnSetROI');
            if (setRoiBtnNow) setRoiBtnNow.style.display = settings.roi_enabled ? 'flex' : 'none';
            const clearRoiBtnNow = document.getElementById('btnClearROI');
            if (clearRoiBtnNow) clearRoiBtnNow.style.display = settings.roi_enabled ? 'flex' : 'none';

            const confirmMsg = document.getElementById('saveConfirm');
            confirmMsg.classList.remove('hidden');
            setTimeout(() => confirmMsg.classList.add('hidden'), 3000);
        });
    }

    // ── ROI capture (Live-page Set-ROI button) ─────────────────────────────
    // 4 clicks on a snapshot → axis-aligned bbox → POST /api/settings.
    // Pauses the WebSocket-to-canvas draw via roiCaptureActive flag.
    (function setupRoiCapture() {
        const btnSetROI = document.getElementById('btnSetROI');
        const banner = document.getElementById('roiCaptureBanner');
        const statusEl = document.getElementById('roiCaptureStatus');
        const buttons = document.getElementById('roiCaptureButtons');
        const btnSave = document.getElementById('btnSaveROI');
        const btnCancel = document.getElementById('btnCancelROI');
        if (!btnSetROI) return;

        // Drag-to-rectangle ROI selection. Replaces the prior 4-click corner picker.
        let dragStart = null;          // {x, y} in native canvas coords
        let pendingBbox = null;        // [x1, y1, x2, y2] in native canvas coords
        let snapshotImg = null;
        let onMouseDown = null, onMouseMove = null, onMouseUp = null;

        function endCapture() {
            roiCaptureActive = false;
            banner.style.display = 'none';
            buttons.style.display = 'none';
            if (onMouseDown) videoCanvas.removeEventListener('pointerdown', onMouseDown);
            if (onMouseMove) videoCanvas.removeEventListener('pointermove', onMouseMove);
            if (onMouseUp)   {
                videoCanvas.removeEventListener('pointerup', onMouseUp);
                videoCanvas.removeEventListener('pointercancel', onMouseUp);
            }
            videoCanvas.style.cursor = '';
            dragStart = null;
            pendingBbox = null;
            snapshotImg = null;
        }

        function clientToCanvas(e) {
            // The canvas has `object-fit: contain` in CSS — its drawing surface
            // is letterboxed inside the CSS box if the aspect ratios differ.
            // getBoundingClientRect() returns the OUTER CSS box including the
            // letterbox bars; subtract those to get the true content area.
            const r = videoCanvas.getBoundingClientRect();
            const cssAspect = r.width / r.height;
            const canvasAspect = videoCanvas.width / videoCanvas.height;
            let contentW, contentH, offX, offY;
            if (canvasAspect > cssAspect) {
                contentW = r.width;
                contentH = r.width / canvasAspect;
                offX = 0;
                offY = (r.height - contentH) / 2;
            } else {
                contentH = r.height;
                contentW = r.height * canvasAspect;
                offX = (r.width - contentW) / 2;
                offY = 0;
            }
            const cssX = Math.max(0, Math.min(contentW, e.clientX - r.left - offX));
            const cssY = Math.max(0, Math.min(contentH, e.clientY - r.top - offY));
            return {
                x: Math.round(cssX * (videoCanvas.width / contentW)),
                y: Math.round(cssY * (videoCanvas.height / contentH))
            };
        }

        function redrawSnapshot(extra) {
            if (!snapshotImg) return;
            vCtx.drawImage(snapshotImg, 0, 0);
            if (extra) extra();
        }

        async function beginCapture() {
            try {
                const res = await fetch('/api/snapshot');
                if (!res.ok) {
                    const msg = (translations[currentLang] && translations[currentLang]['roi_need_stream'])
                                || 'Start the stream first to capture a snapshot.';
                    alert(msg);
                    return;
                }
                const blob = await res.blob();
                const img = new Image();
                img.onload = () => {
                    roiCaptureActive = true;
                    snapshotImg = img;
                    videoCanvas.width = img.naturalWidth;
                    videoCanvas.height = img.naturalHeight;
                    vCtx.drawImage(img, 0, 0);
                    videoCanvas.style.cursor = 'crosshair';
                    dragStart = null;
                    pendingBbox = null;
                    banner.style.display = 'block';
                    buttons.style.display = 'none';
                    statusEl.textContent = (translations[currentLang] && translations[currentLang]['roi_drag_instruction'])
                                           || 'Click and drag a rectangle over the conveyor belt area.';

                    // Use pointer events with setPointerCapture so the drag keeps
                    // updating even when the cursor leaves the canvas — only ends
                    // when the button is actually released.
                    onMouseDown = (e) => {
                        if (e.button !== 0) return;
                        e.preventDefault();
                        try { videoCanvas.setPointerCapture(e.pointerId); } catch (_) {}
                        dragStart = clientToCanvas(e);
                        pendingBbox = null;
                        buttons.style.display = 'none';
                    };

                    onMouseMove = (e) => {
                        if (!dragStart) return;
                        e.preventDefault();
                        const cur = clientToCanvas(e);
                        redrawSnapshot(() => {
                            const lw = Math.max(2, Math.round(videoCanvas.width / 400));
                            vCtx.strokeStyle = '#00ff00';
                            vCtx.lineWidth = lw;
                            vCtx.setLineDash([Math.max(6, lw * 3), Math.max(4, lw * 2)]);
                            vCtx.strokeRect(
                                Math.min(dragStart.x, cur.x),
                                Math.min(dragStart.y, cur.y),
                                Math.abs(cur.x - dragStart.x),
                                Math.abs(cur.y - dragStart.y)
                            );
                            vCtx.setLineDash([]);
                        });
                    };

                    onMouseUp = (e) => {
                        if (!dragStart) return;
                        e.preventDefault();
                        try { videoCanvas.releasePointerCapture(e.pointerId); } catch (_) {}
                        const end = clientToCanvas(e);
                        const x1 = Math.min(dragStart.x, end.x);
                        const y1 = Math.min(dragStart.y, end.y);
                        const x2 = Math.max(dragStart.x, end.x);
                        const y2 = Math.max(dragStart.y, end.y);
                        if (x2 - x1 < 20 || y2 - y1 < 20) {
                            redrawSnapshot();
                            dragStart = null;
                            return;
                        }
                        pendingBbox = [x1, y1, x2, y2];
                        redrawSnapshot(() => {
                            const lw = Math.max(2, Math.round(videoCanvas.width / 400));
                            vCtx.strokeStyle = '#00ff00';
                            vCtx.lineWidth = lw;
                            vCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                            vCtx.fillStyle = 'rgba(0, 255, 0, 0.18)';
                            vCtx.fillRect(x1, y1, x2 - x1, y2 - y1);
                        });
                        statusEl.textContent =
                            `Captured: x=[${x1},${x2}] y=[${y1},${y2}] (${x2-x1}×${y2-y1}) — drag again to redraw, or save.`;
                        buttons.style.display = 'flex';
                        dragStart = null;
                    };

                    videoCanvas.addEventListener('pointerdown', onMouseDown);
                    videoCanvas.addEventListener('pointermove', onMouseMove);
                    videoCanvas.addEventListener('pointerup', onMouseUp);
                    videoCanvas.addEventListener('pointercancel', onMouseUp);
                };
                img.src = URL.createObjectURL(blob);
            } catch (e) {
                console.warn('ROI capture failed', e);
                alert('Could not start ROI capture: ' + e);
            }
        }

        btnSetROI.addEventListener('click', beginCapture);
        btnCancel.addEventListener('click', endCapture);

        // Clear ROI: wipes the saved bbox so the next stream session starts in
        // full-frame mode. Operator can then re-draw with Set ROI. Sends an
        // empty `roi_points: []` (the backend treats <4 points as no ROI) and
        // keeps `roi_enabled: true` so the Set/Clear buttons stay visible.
        const btnClearROI = document.getElementById('btnClearROI');
        if (btnClearROI) {
            btnClearROI.addEventListener('click', async () => {
                try {
                    const res = await fetch('/api/settings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', ...devHeaders() },
                        body: JSON.stringify({ roi_enabled: true, roi_points: [] })
                    });
                    const body = await res.json().catch(() => ({}));
                    if (!res.ok || body.status !== 'success') {
                        const prefix = (translations[currentLang] && translations[currentLang]['roi_clear_failed']) || 'Could not clear ROI: ';
                        alert(prefix + (body.message || `HTTP ${res.status}`));
                        return;
                    }
                    const ok = (translations[currentLang] && translations[currentLang]['roi_cleared_alert'])
                               || 'ROI cleared. Stop and Start the stream to apply.';
                    alert(ok);
                    // Reflect the cleared state in the Settings → Camera readout.
                    const roiCurrentEl = document.getElementById('disp_roi_current');
                    if (roiCurrentEl) {
                        roiCurrentEl.setAttribute('data-i18n', 'roi_none');
                        roiCurrentEl.textContent = (translations[currentLang] && translations[currentLang]['roi_none']) || 'none (full frame)';
                    }
                } catch (e) {
                    console.warn('ROI clear failed', e);
                    alert('Could not clear ROI: ' + e);
                }
            });
        }

        btnSave.addEventListener('click', async () => {
            if (!pendingBbox) { endCapture(); return; }
            const [x1, y1, x2, y2] = pendingBbox;
            const points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]];
            try {
                const res = await fetch('/api/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...devHeaders() },
                    body: JSON.stringify({ roi_enabled: true, roi_points: points })
                });
                const body = await res.json().catch(() => ({}));
                if (!res.ok || body.status !== 'success') {
                    const prefix = (translations[currentLang] && translations[currentLang]['roi_save_failed']) || 'Could not save ROI: ';
                    alert(prefix + (body.message || `HTTP ${res.status}`));
                    return;
                }
                const ok = (translations[currentLang] && translations[currentLang]['roi_saved'])
                           || 'ROI saved. Stop and Start the stream to apply.';
                alert(ok);
                const roiCurrentEl = document.getElementById('disp_roi_current');
                if (roiCurrentEl) {
                    const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
                    const x1 = Math.min(...xs), x2 = Math.max(...xs);
                    const y1 = Math.min(...ys), y2 = Math.max(...ys);
                    roiCurrentEl.removeAttribute('data-i18n');
                    roiCurrentEl.textContent = `x=[${x1},${x2}] y=[${y1},${y2}] (${x2-x1}×${y2-y1})`;
                }
            } catch (e) {
                console.warn('ROI save failed', e);
                alert('Could not save ROI: ' + e);
            } finally {
                endCapture();
            }
        });
    })();

    loadSettings();

    // French is the default — translate DOM text on load so users don't see
    // an English flash. Users can switch to English via the header button.
    // Placed at the very end: setLanguage() touches `currentSourceType`
    // (declared inside this handler with `let`), so calling it earlier hits
    // the temporal dead zone and throws, aborting the entire init block.
    setLanguage('fr');

});
