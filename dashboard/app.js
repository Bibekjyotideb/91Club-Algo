/**
 * Win Go Predictor — Dashboard Client v2
 * Real-time WebSocket + API interaction
 * Per-timer prediction state, auto-refresh on tab switch
 */

// ============ STATE ============
let ws = null;
let reconnectTimer = null;
let autoPredict = true;
let resultCount = 0;
let activeTimer = '3min';  // currently selected timer view

// Per-timer prediction cache
let timerPredictions = {
    '30sec': null,
    '1min': null,
    '3min': null,
};

// ============ DOM REFS ============
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ============ WEBSOCKET ============
let pingInterval = null;

function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws`);

    ws.onopen = () => {
        updateConnectionStatus('connected');
        addLog('Connected to server');

        // Start keep-alive ping
        if (pingInterval) clearInterval(pingInterval);
        pingInterval = setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    };

    ws.onclose = () => {
        updateConnectionStatus('disconnected');
        addLog('Disconnected. Reconnecting...');
        if (pingInterval) clearInterval(pingInterval);
        reconnectTimer = setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = () => {
        updateConnectionStatus('disconnected');
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleMessage(msg);
    };
}

function handleMessage(msg) {
    switch (msg.type) {
        case 'result':
            // Always cache, only render for active timer
            if (msg.data.timer === activeTimer || !msg.data.timer) {
                onNewResult(msg.data);
            }
            break;

        case 'prediction':
            // Cache prediction for its timer
            const predTimer = msg.data.timer || '3min';
            timerPredictions[predTimer] = msg.data;

            // Only render if it matches the active timer
            if (predTimer === activeTimer) {
                renderPrediction(msg.data);
            }
            break;

        case 'stats':
            const statsTimer = msg.data.timer || 'all';
            if (statsTimer === activeTimer || statsTimer === 'all') {
                onStats(msg.data);
            }
            break;

        case 'history':
            onHistory(msg.data);
            break;

        case 'timer_counts':
            updateTimerCounts(msg.data);
            break;

        case 'predictions_history':
            renderPredictionHistory(msg.data || []);
            break;

        case 'bulk_complete':
            addLog(`[${msg.timer || '?'}] Bulk: ${msg.count} results. Training: ${msg.training?.status || 'done'}`);
            fetchStats();
            break;

        case 'training_complete':
            $('#trainingStatus').style.display = 'none';
            addLog(`[${msg.timer || '?'}] Training complete: accuracy ${msg.result?.best_val_accuracy || '?'}%`);
            break;
    }
}

// ============ TIMER SWITCHING ============
function switchTimer(timer) {
    if (timer === activeTimer) return;  // no-op if already selected

    activeTimer = timer;

    // Update tab UI
    $$('.timer-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.timer === timer);
    });

    // Clear current view and show loading state
    $('#resultsFeed').innerHTML = '<div class="empty-state"><p>Loading...</p></div>';
    chartLabels.length = 0;
    chartAccData.length = 0;
    chartRecentData.length = 0;
    if (accuracyChart) accuracyChart.update('none');

    // Immediately show cached prediction for this timer (or clear)
    if (timerPredictions[timer]) {
        renderPrediction(timerPredictions[timer]);
    } else {
        clearPredictionCard();
    }

    // Request data for this timer via WebSocket
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'switch_timer', timer: timer }));
    }

    // Also fetch fresh data via REST
    fetchStats();
    fetchPredictionHistory();
    fetchFreshPrediction(timer);

    addLog(`Switched to ${timer} timer`);
}

function clearPredictionCard() {
    $('#predictionLabel').textContent = '—';
    $('#predictionLabel').className = 'prediction-label';
    $('#predictionCard').className = 'prediction-card';
    $('#confValue').textContent = '—';

    // Reset confidence ring
    const circumference = 2 * Math.PI * 52;
    $('#ringFill').style.strokeDashoffset = circumference;
    $('#ringFill').style.stroke = 'var(--accent-blue)';

    // Reset model bars
    ['lstmBar', 'markovBar', 'patternBar', 'freqBar'].forEach(id => {
        const el = $(`#${id}`);
        if (el) el.style.width = '50%';
    });
    ['lstmPct', 'markovPct', 'patternPct', 'freqPct'].forEach(id => {
        const el = $(`#${id}`);
        if (el) el.textContent = '—';
    });
}

async function fetchFreshPrediction(timer) {
    try {
        const resp = await fetch(`/api/predict?timer=${timer}`);
        const data = await resp.json();

        if (data.error) {
            // Not enough data for this timer
            addLog(`[${timer}] ${data.error} (${data.count || 0} results)`);
            return;
        }

        // Cache and render if still on this timer
        timerPredictions[timer] = data;
        if (activeTimer === timer) {
            renderPrediction(data);
        }
    } catch (e) {
        console.error('Failed to fetch prediction:', e);
    }
}

function updateTimerCounts(counts) {
    for (const [timer, count] of Object.entries(counts)) {
        const el = $(`#count-${timer}`);
        if (el) el.textContent = count;
    }
}

// ============ CONNECTION STATUS ============
function updateConnectionStatus(status) {
    const dot = $('.status-dot');
    const text = $('.status-text');

    dot.className = 'status-dot';
    if (status === 'connected') {
        dot.classList.add('connected');
        text.textContent = 'Live';
    } else if (status === 'disconnected') {
        dot.classList.add('disconnected');
        text.textContent = 'Offline';
    } else {
        text.textContent = 'Connecting...';
    }
}

// ============ RESULT HANDLING ============
function onNewResult(data) {
    addResultPill(data);
    resultCount++;
    $('#resultCount').textContent = `${resultCount} results`;
    addLog(`Result: ${data.digit} (${data.label.toUpperCase()})${data.prediction_correct !== undefined ? (data.prediction_correct ? ' ✓' : ' ✗') : ''}`);
}

function addResultPill(data) {
    const feed = $('#resultsFeed');
    const empty = feed.querySelector('.empty-state');
    if (empty) empty.remove();

    const pill = document.createElement('div');
    pill.className = `result-pill ${data.label}`;
    pill.textContent = data.digit;
    pill.title = `Round: ${data.round_id || '?'}\nLabel: ${data.label}\nTime: ${new Date(data.timestamp * 1000).toLocaleTimeString()}`;

    feed.insertBefore(pill, feed.firstChild);

    // Keep max 200 pills
    while (feed.children.length > 200) {
        feed.removeChild(feed.lastChild);
    }
}

function onHistory(data) {
    const feed = $('#resultsFeed');
    feed.innerHTML = '';

    if (!data || data.length === 0) return;

    resultCount = data.length;
    $('#resultCount').textContent = `${resultCount} results`;

    data.forEach(item => {
        const pill = document.createElement('div');
        pill.className = `result-pill ${item.label}`;
        pill.textContent = item.digit;
        feed.appendChild(pill);
    });
}

// ============ PREDICTION RENDERING ============
function renderPrediction(data) {
    const card = $('#predictionCard');
    const label = $('#predictionLabel');
    const confValue = $('#confValue');
    const ringFill = $('#ringFill');

    // Update label
    label.textContent = data.prediction.toUpperCase();
    label.className = `prediction-label ${data.prediction}`;

    // Update card glow
    card.className = `prediction-card ${data.prediction}-prediction`;

    // Update confidence ring
    const confPct = Math.round(data.confidence * 100);
    confValue.textContent = `${confPct}%`;

    const circumference = 2 * Math.PI * 52;
    const offset = circumference * (1 - data.confidence);
    ringFill.style.strokeDashoffset = offset;
    ringFill.style.stroke = data.prediction === 'big'
        ? 'var(--accent-big)'
        : 'var(--accent-small)';

    // Update model breakdown
    if (data.models) {
        updateModelBar('lstm', data.models.lstm);
        updateModelBar('markov', data.models.markov);
        updateModelBar('pattern', data.models.pattern);
        updateModelBar('frequency', data.models.frequency);
    }

    // Show timer badge
    const badge = $('#modelBadge');
    if (badge && data.timer) {
        badge.textContent = `${data.timer} · Ensemble`;
    }

    addLog(`[${data.timer || '?'}] Prediction: ${data.prediction.toUpperCase()} (${confPct}% conf)`);
}

function updateModelBar(name, modelData) {
    if (!modelData) return;

    const barId = name === 'frequency' ? 'freqBar' : `${name}Bar`;
    const pctId = name === 'frequency' ? 'freqPct' : `${name}Pct`;

    const bar = $(`#${barId}`);
    const pct = $(`#${pctId}`);

    if (bar) {
        const bigPct = Math.round(modelData.prob_big * 100);
        bar.style.width = `${bigPct}%`;
    }
    if (pct) {
        pct.textContent = `${Math.round(modelData.prob_big * 100)}%`;
    }
}

// ============ STATS ============
function onStats(data) {
    $('#overallAccuracy').textContent = `${data.overall_accuracy || 0}%`;
    $('#recentAccuracy').textContent = `${data.last_20_accuracy || 0}%`;
    $('#totalResults').textContent = data.total_results || 0;
    $('#totalPredictions').textContent = data.total_predictions || 0;

    // Update accuracy chart
    updateAccuracyChart(data);

    // Refresh prediction history
    fetchPredictionHistory();
}

async function fetchStats() {
    try {
        const resp = await fetch(`/api/stats?timer=${activeTimer}`);
        const data = await resp.json();
        onStats(data);

        if (data.timer_counts) {
            updateTimerCounts(data.timer_counts);
        }
    } catch (e) {
        console.error('Failed to fetch stats:', e);
    }
}

// ============ ACCURACY CHART ============
let accuracyChart = null;
let chartLabels = [];
let chartAccData = [];
let chartRecentData = [];

function initChart() {
    const ctx = $('#accuracyChart').getContext('2d');

    accuracyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartLabels,
            datasets: [
                {
                    label: 'Overall %',
                    data: chartAccData,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 2,
                    pointHoverRadius: 5
                },
                {
                    label: 'Last 20 %',
                    data: chartRecentData,
                    borderColor: '#22c55e',
                    backgroundColor: 'rgba(34, 197, 94, 0.05)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 2,
                    pointHoverRadius: 5
                },
                {
                    label: '50% Baseline',
                    data: [],
                    borderColor: 'rgba(239, 68, 68, 0.4)',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#94a3b8',
                        font: { family: 'Inter', size: 11 },
                        boxWidth: 12,
                        padding: 16
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { color: '#64748b', font: { size: 10 }, maxTicksLimit: 10 }
                },
                y: {
                    min: 0,
                    max: 100,
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: {
                        color: '#64748b',
                        font: { size: 10 },
                        callback: v => v + '%'
                    }
                }
            }
        }
    });
}

function updateAccuracyChart(stats) {
    const now = new Date().toLocaleTimeString();
    chartLabels.push(now);
    chartAccData.push(stats.overall_accuracy || 0);
    chartRecentData.push(stats.last_20_accuracy || 0);

    // Update baseline
    accuracyChart.data.datasets[2].data = chartLabels.map(() => 50);

    // Keep last 50 points
    if (chartLabels.length > 50) {
        chartLabels.shift();
        chartAccData.shift();
        chartRecentData.shift();
    }

    accuracyChart.update('none');
}

// ============ PREDICTION HISTORY TABLE ============
async function fetchPredictionHistory() {
    try {
        const resp = await fetch(`/api/predictions?limit=30&timer=${activeTimer}`);
        const data = await resp.json();
        renderPredictionHistory(data.predictions || []);
    } catch (e) {
        console.error('Failed to fetch predictions:', e);
    }
}

function renderPredictionHistory(predictions) {
    const tbody = $('#historyBody');
    tbody.innerHTML = '';

    predictions.forEach((pred, i) => {
        const tr = document.createElement('tr');

        const resultTag = pred.correct === 1
            ? '<span class="tag correct">✓ HIT</span>'
            : pred.correct === 0
                ? '<span class="tag wrong">✗ MISS</span>'
                : '<span class="tag pending">PENDING</span>';

        const timerBadge = pred.timer ? `<span class="tag timer-tag">${pred.timer}</span>` : '';

        tr.innerHTML = `
            <td>${predictions.length - i}</td>
            <td><span class="tag ${pred.predicted_label}">${pred.predicted_label.toUpperCase()}</span></td>
            <td>${Math.round(pred.confidence * 100)}%</td>
            <td>${pred.actual_label ? `<span class="tag ${pred.actual_label}">${pred.actual_label.toUpperCase()}</span>` : '—'}</td>
            <td>${resultTag}</td>
        `;
        tbody.appendChild(tr);
    });
}

// ============ INPUT HANDLERS ============
async function submitDigit(digit) {
    try {
        const resp = await fetch('/api/result', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ digit, source: 'manual', timer: activeTimer })
        });
        const data = await resp.json();

        if (data.status === 'ok') {
            // result and prediction will come via WebSocket
        }
    } catch (e) {
        addLog(`Error submitting digit: ${e.message}`);
    }
}

async function submitBulk() {
    const input = $('#bulkInput');
    const raw = input.value.trim();
    if (!raw) return;

    const digits = raw.split(/[\s,;]+/).map(Number).filter(d => d >= 0 && d <= 9);

    if (digits.length === 0) {
        addLog('Invalid bulk input. Use digits 0-9 separated by spaces.');
        return;
    }

    addLog(`Submitting ${digits.length} results for ${activeTimer}...`);
    $('#trainingStatus').style.display = 'flex';

    try {
        const resp = await fetch('/api/results/bulk', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ digits, timer: activeTimer })
        });
        const data = await resp.json();

        input.value = '';
        addLog(`[${activeTimer}] Bulk complete: ${data.count} results. Val accuracy: ${data.training?.best_val_accuracy || '?'}%`);
        $('#trainingStatus').style.display = 'none';

        // Refresh everything
        fetchStats();
        location.reload();
    } catch (e) {
        addLog(`Bulk error: ${e.message}`);
        $('#trainingStatus').style.display = 'none';
    }
}

// ============ CONTROL HANDLERS ============
async function triggerTrain() {
    addLog(`Starting ${activeTimer} model training...`);
    $('#trainingStatus').style.display = 'flex';

    try {
        const resp = await fetch(`/api/train?timer=${activeTimer}`, { method: 'POST' });
        const data = await resp.json();
        addLog(`[${activeTimer}] Training result: ${JSON.stringify(data)}`);
    } catch (e) {
        addLog(`Training error: ${e.message}`);
    }

    $('#trainingStatus').style.display = 'none';
}

async function triggerPredict() {
    try {
        const resp = await fetch(`/api/predict?timer=${activeTimer}`);
        const data = await resp.json();

        if (data.error) {
            addLog(`Prediction error: ${data.error}`);
        } else {
            data.timer = data.timer || activeTimer;
            timerPredictions[activeTimer] = data;
            renderPrediction(data);
        }
    } catch (e) {
        addLog(`Predict error: ${e.message}`);
    }
}

async function resetModel() {
    if (!confirm(`Reset the ${activeTimer} model? This will re-initialize all weights.`)) return;

    try {
        await fetch(`/api/reset?timer=${activeTimer}`, { method: 'POST' });
        addLog(`[${activeTimer}] Model reset. Retrain recommended.`);
        timerPredictions[activeTimer] = null;
        clearPredictionCard();
    } catch (e) {
        addLog(`Reset error: ${e.message}`);
    }
}

async function toggleAutoPredict() {
    try {
        const resp = await fetch('/api/toggle-auto-predict', { method: 'POST' });
        const data = await resp.json();
        autoPredict = data.auto_predict;

        const btn = $('#btnAutoPredict');
        btn.textContent = `Auto-Predict: ${autoPredict ? 'ON' : 'OFF'}`;
        btn.classList.toggle('active', autoPredict);

        addLog(`Auto-predict: ${autoPredict ? 'enabled' : 'disabled'}`);
    } catch (e) {
        addLog(`Toggle error: ${e.message}`);
    }
}

// ============ LOGGING ============
function addLog(message) {
    const box = $('#logBox');
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    const time = new Date().toLocaleTimeString();
    entry.textContent = `[${time}] ${message}`;
    box.insertBefore(entry, box.firstChild);

    // Keep last 50 entries
    while (box.children.length > 50) {
        box.removeChild(box.lastChild);
    }
}

// ============ INIT ============
document.addEventListener('DOMContentLoaded', () => {
    // Init chart
    initChart();

    // Connect WebSocket
    connectWebSocket();

    // Digit buttons
    $$('.digit-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const digit = parseInt(btn.dataset.digit);
            submitDigit(digit);
        });
    });

    // Bulk input
    $('#bulkSubmit').addEventListener('click', submitBulk);
    $('#bulkInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') submitBulk();
    });

    // Controls
    $('#btnTrain').addEventListener('click', triggerTrain);
    $('#btnPredict').addEventListener('click', triggerPredict);
    $('#btnReset').addEventListener('click', resetModel);
    $('#btnAutoPredict').addEventListener('click', toggleAutoPredict);

    // Timer tabs
    $$('.timer-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            switchTimer(tab.dataset.timer);
        });
    });

    // Initial data fetch
    fetchStats();
    fetchPredictionHistory();
    // Fetch initial prediction for default timer
    fetchFreshPrediction(activeTimer);
});
