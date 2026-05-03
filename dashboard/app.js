/**
 * Win Go Predictor — Dashboard Client v3
 * Real-time WebSocket + API interaction
 * Per-timer prediction state — Number · Color · Size
 */

// ============ STATE ============
let ws = null;
let reconnectTimer = null;
let autoPredict = true;
let resultCount = 0;
let activeTimer = '1min';

let timerPredictions = { '30sec': null, '1min': null, '3min': null };

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
        if (pingInterval) clearInterval(pingInterval);
        pingInterval = setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN)
                ws.send(JSON.stringify({ type: 'ping' }));
        }, 30000);
    };
    ws.onclose = () => {
        updateConnectionStatus('disconnected');
        addLog('Disconnected. Reconnecting...');
        if (pingInterval) clearInterval(pingInterval);
        reconnectTimer = setTimeout(connectWebSocket, 3000);
    };
    ws.onerror = () => updateConnectionStatus('disconnected');
    ws.onmessage = (event) => handleMessage(JSON.parse(event.data));
}

function handleMessage(msg) {
    switch (msg.type) {
        case 'result':
            if (msg.data.timer === activeTimer || !msg.data.timer) onNewResult(msg.data);
            break;
        case 'prediction':
            const predTimer = msg.data.timer || '3min';
            timerPredictions[predTimer] = msg.data;
            if (predTimer === activeTimer) renderPrediction(msg.data);
            break;
        case 'stats':
            const statsTimer = msg.data.timer || 'all';
            if (statsTimer === activeTimer || statsTimer === 'all') onStats(msg.data);
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
            addLog(`[${msg.timer || '?'}] Bulk: ${msg.count} results.`);
            fetchStats();
            break;
        case 'training_complete':
            $('#trainingStatus').style.display = 'none';
            addLog(`[${msg.timer || '?'}] Training complete`);
            break;
    }
}

// ============ TIMER SWITCHING ============
function switchTimer(timer) {
    if (timer === activeTimer) return;
    activeTimer = timer;
    $$('.timer-tab').forEach(tab => tab.classList.toggle('active', tab.dataset.timer === timer));
    $('#resultsFeed').innerHTML = '<div class="empty-state"><p>Loading...</p></div>';
    chartLabels.length = 0; chartAccData.length = 0; chartRecentData.length = 0;
    if (accuracyChart) accuracyChart.update('none');
    if (timerPredictions[timer]) renderPrediction(timerPredictions[timer]);
    else clearAllPredictions();
    if (ws && ws.readyState === WebSocket.OPEN)
        ws.send(JSON.stringify({ type: 'switch_timer', timer }));
    fetchStats(); fetchPredictionHistory(); fetchFreshPrediction(timer);
    addLog(`Switched to ${timer} timer`);
}

function clearAllPredictions() {
    // Size
    $('#predictionLabel').textContent = '—';
    $('#predictionLabel').className = 'prediction-label';
    $('#predictionCard').className = 'pred-card prediction-card';
    $('#confValue').textContent = '—';
    const circumference = 2 * Math.PI * 52;
    $('#ringFill').style.strokeDashoffset = circumference;
    ['lstmBar','markovBar','patternBar','freqBar'].forEach(id => { const el = $(`#${id}`); if(el) el.style.width='50%'; });
    ['lstmPct','markovPct','patternPct','freqPct'].forEach(id => { const el = $(`#${id}`); if(el) el.textContent='—'; });
    // Number
    $('#predictedNumber').textContent = '—';
    $('#numberConf').textContent = '—';
    for (let i = 0; i < 10; i++) {
        const fill = $(`#prob-${i}`); if (fill) fill.style.width = '10%';
        const pct = $(`#probPct-${i}`); if (pct) pct.textContent = '—';
    }
    // Color
    $('#predictedColor').textContent = '—';
    $('#predictedColor').className = 'predicted-color';
    $('#colorPredCard').className = 'pred-card color-pred-card';
    $('#colorConf').textContent = '—';
    ['Red','Green','Violet'].forEach(c => {
        const bar = $(`#colorBar${c}`); if(bar) bar.style.width = '33%';
        const pct = $(`#colorPct${c}`); if(pct) pct.textContent = '—';
    });
}

async function fetchFreshPrediction(timer) {
    try {
        const resp = await fetch(`/api/predict?timer=${timer}`);
        const data = await resp.json();
        if (data.error) { addLog(`[${timer}] ${data.error}`); return; }
        timerPredictions[timer] = data;
        if (activeTimer === timer) renderPrediction(data);
    } catch (e) { console.error('Prediction fetch failed:', e); }
}

function updateTimerCounts(counts) {
    for (const [timer, count] of Object.entries(counts)) {
        const el = $(`#count-${timer}`); if (el) el.textContent = count;
    }
}

// ============ CONNECTION STATUS ============
function updateConnectionStatus(status) {
    const dot = $('.status-dot');
    const text = $('.status-text');
    dot.className = 'status-dot';
    if (status === 'connected') { dot.classList.add('connected'); text.textContent = 'Live'; }
    else if (status === 'disconnected') { dot.classList.add('disconnected'); text.textContent = 'Offline'; }
    else text.textContent = 'Connecting...';
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
    const empty = feed.querySelector('.empty-state'); if (empty) empty.remove();
    const pill = document.createElement('div');
    pill.className = `result-pill ${digitColorClass(data.digit)}`;
    pill.textContent = data.digit;
    pill.title = `Round: ${data.round_id || '?'}\nLabel: ${data.label}\nColor: ${data.color || '?'}\nTime: ${new Date(data.timestamp * 1000).toLocaleTimeString()}`;
    feed.insertBefore(pill, feed.firstChild);
    while (feed.children.length > 200) feed.removeChild(feed.lastChild);
}

function onHistory(data) {
    const feed = $('#resultsFeed'); feed.innerHTML = '';
    if (!data || data.length === 0) return;
    resultCount = data.length;
    $('#resultCount').textContent = `${resultCount} results`;
    data.forEach(item => {
        const pill = document.createElement('div');
        pill.className = `result-pill ${digitColorClass(item.digit)}`;
        pill.textContent = item.digit;
        feed.appendChild(pill);
    });
}

// ============ PREDICTION RENDERING ============
function formatPeriod(roundId) {
    if (!roundId || roundId === 'null') return '--';
    return roundId.slice(-5);
}

function renderPrediction(data) {
    renderSizePrediction(data);
    if (data.number) renderNumberPrediction(data.number);
    if (data.color) renderColorPrediction(data.color);
    // Update period on all cards
    const period = formatPeriod(data.for_round);
    const periodText = `Period: ${period}`;
    const np = $('#numberPeriod'); if (np) np.textContent = periodText;
    const cp = $('#colorPeriod'); if (cp) cp.textContent = periodText;
    const sp = $('#sizePeriod'); if (sp) sp.textContent = periodText;
}

function renderSizePrediction(data) {
    const card = $('#predictionCard');
    const label = $('#predictionLabel');
    const confValue = $('#confValue');
    const ringFill = $('#ringFill');

    label.textContent = data.prediction.toUpperCase();
    label.className = `prediction-label ${data.prediction}`;
    card.className = `pred-card prediction-card ${data.prediction}-prediction`;

    const confPct = Math.round(data.confidence * 100);
    confValue.textContent = `${confPct}%`;
    const circumference = 2 * Math.PI * 52;
    ringFill.style.strokeDashoffset = circumference * (1 - data.confidence);
    ringFill.style.stroke = data.prediction === 'big' ? 'var(--accent-big)' : 'var(--accent-small)';

    if (data.models) {
        updateModelBar('lstm', data.models.lstm);
        updateModelBar('markov', data.models.markov);
        updateModelBar('pattern', data.models.pattern);
        updateModelBar('frequency', data.models.frequency);
    }

    const badge = $('#modelBadge');
    if (badge && data.timer) badge.textContent = `${data.timer} · Ensemble`;

    addLog(`[${data.timer || '?'}] Size: ${data.prediction.toUpperCase()} (${confPct}%)`);
}

function renderNumberPrediction(numData) {
    const display = $('#predictedNumber');
    const conf = $('#numberConf');
    display.textContent = numData.predicted;
    const confPct = Math.round(numData.confidence * 100);
    conf.textContent = `${confPct}% confidence`;

    // Highlight top pick
    const probs = numData.probabilities || [];
    const maxProb = Math.max(...probs);

    for (let i = 0; i < 10; i++) {
        const p = probs[i] || 0.1;
        const pct = Math.round(p * 100);
        const fill = $(`#prob-${i}`);
        const pctEl = $(`#probPct-${i}`);
        const row = fill?.closest('.prob-bar-row');
        if (fill) fill.style.width = `${Math.max(pct, 2)}%`;
        if (pctEl) pctEl.textContent = `${pct}%`;
        if (row) row.classList.toggle('top-pick', p === maxProb);
    }

    addLog(`[${activeTimer}] Number: ${numData.predicted} (${confPct}%)`);
}

function renderColorPrediction(colorData) {
    const display = $('#predictedColor');
    const conf = $('#colorConf');
    const card = $('#colorPredCard');

    display.textContent = colorData.predicted.toUpperCase();
    display.className = `predicted-color ${colorData.predicted}`;
    card.className = `pred-card color-pred-card ${colorData.predicted}-prediction`;
    const confPct = Math.round(colorData.confidence * 100);
    conf.textContent = `${confPct}% confidence`;

    const probs = colorData.probabilities || {};
    const redPct = Math.round((probs.red || 0) * 100);
    const greenPct = Math.round((probs.green || 0) * 100);
    const violetPct = Math.round((probs.violet || 0) * 100);

    const barR = $('#colorBarRed'); if (barR) barR.style.width = `${redPct}%`;
    const barG = $('#colorBarGreen'); if (barG) barG.style.width = `${greenPct}%`;
    const barV = $('#colorBarViolet'); if (barV) barV.style.width = `${violetPct}%`;
    const pctR = $('#colorPctRed'); if (pctR) pctR.textContent = `${redPct}%`;
    const pctG = $('#colorPctGreen'); if (pctG) pctG.textContent = `${greenPct}%`;
    const pctV = $('#colorPctViolet'); if (pctV) pctV.textContent = `${violetPct}%`;

    addLog(`[${activeTimer}] Color: ${colorData.predicted.toUpperCase()} (${confPct}%)`);
}

function updateModelBar(name, modelData) {
    if (!modelData) return;
    const barId = name === 'frequency' ? 'freqBar' : `${name}Bar`;
    const pctId = name === 'frequency' ? 'freqPct' : `${name}Pct`;
    const bar = $(`#${barId}`); const pct = $(`#${pctId}`);
    if (bar) bar.style.width = `${Math.round(modelData.prob_big * 100)}%`;
    if (pct) pct.textContent = `${Math.round(modelData.prob_big * 100)}%`;
}

// ============ STATS ============
function onStats(data) {
    $('#overallAccuracy').textContent = `${data.overall_accuracy || 0}%`;
    $('#recentAccuracy').textContent = `${data.last_20_accuracy || 0}%`;
    $('#totalResults').textContent = data.total_results || 0;
    $('#totalPredictions').textContent = data.total_predictions || 0;
    updateAccuracyChart(data);
    fetchPredictionHistory();
}

async function fetchStats() {
    try {
        const resp = await fetch(`/api/stats?timer=${activeTimer}`);
        const data = await resp.json();
        onStats(data);
        if (data.timer_counts) updateTimerCounts(data.timer_counts);
    } catch (e) { console.error('Stats fetch failed:', e); }
}

// ============ CHART ============
let accuracyChart = null, chartLabels = [], chartAccData = [], chartRecentData = [];

function initChart() {
    const ctx = $('#accuracyChart').getContext('2d');
    accuracyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartLabels,
            datasets: [
                { label: 'Overall %', data: chartAccData, borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', borderWidth: 2, fill: true, tension: 0.4, pointRadius: 2 },
                { label: 'Last 20 %', data: chartRecentData, borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,0.05)', borderWidth: 2, fill: true, tension: 0.4, pointRadius: 2 },
                { label: '50% Baseline', data: [], borderColor: 'rgba(239,68,68,0.4)', borderWidth: 1, borderDash: [5,5], fill: false, pointRadius: 0 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: { legend: { labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 }, boxWidth: 12, padding: 16 } } },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#64748b', font: { size: 10 }, maxTicksLimit: 10 } },
                y: { min: 0, max: 100, grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#64748b', font: { size: 10 }, callback: v => v + '%' } }
            }
        }
    });
}

function updateAccuracyChart(stats) {
    chartLabels.push(new Date().toLocaleTimeString());
    chartAccData.push(stats.overall_accuracy || 0);
    chartRecentData.push(stats.last_20_accuracy || 0);
    accuracyChart.data.datasets[2].data = chartLabels.map(() => 50);
    if (chartLabels.length > 50) { chartLabels.shift(); chartAccData.shift(); chartRecentData.shift(); }
    accuracyChart.update('none');
}

// ============ PREDICTION HISTORY ============
async function fetchPredictionHistory() {
    try {
        const resp = await fetch(`/api/predictions?limit=30&timer=${activeTimer}`);
        const data = await resp.json();
        renderPredictionHistory(data.predictions || []);
    } catch (e) { console.error('Predictions fetch failed:', e); }
}

/**
 * Derive CSS color class from a digit based on WinGo rules:
 *   Even (0,2,4,6,8) = 'red-num', Odd (1,3,5,7,9) = 'green-num'
 *   0 and 5 additionally get 'violet-ring'
 */
function digitColorClass(digit) {
    const d = parseInt(digit);
    if (isNaN(d) || d < 0) return '';
    const colorClass = (d % 2 === 0) ? 'red-num' : 'green-num';
    if (d === 0 || d === 5) return colorClass + ' violet-ring';
    return colorClass;
}

function getNumberColorClass(digit, color) {
    return digitColorClass(digit);
}

function buildPredBadge(num, color, size) {
    const d = parseInt(num);
    const hasNum = !isNaN(d) && d >= 0;
    const colorClass = getNumberColorClass(num, color);
    const sizeLabel = size ? size.toUpperCase() : '';
    const colorLabel = color ? color.charAt(0).toUpperCase() + color.slice(1) : '';

    if (!hasNum && !sizeLabel) return '<span class="pred-badge empty">--</span>';

    return `<div class="pred-badge-wrap">
        <span class="pred-num ${colorClass}">${hasNum ? d : '?'}</span>
        <span class="pred-meta">${colorLabel} ${sizeLabel}</span>
    </div>`;
}

function renderPredictionHistory(predictions) {
    // Desktop table
    const tbody = $('#historyBody'); tbody.innerHTML = '';
    // Mobile cards
    const cards = $('#historyCards'); if (cards) cards.innerHTML = '';

    predictions.forEach((pred) => {
        const period = formatPeriod(pred.round_id);
        const resultTag = pred.correct === 1 ? '<span class="tag correct">HIT</span>'
            : pred.correct === 0 ? '<span class="tag wrong">MISS</span>'
            : '<span class="tag pending">--</span>';

        const predBadge = buildPredBadge(pred.predicted_number, pred.predicted_color, pred.predicted_label);
        const actDigit = pred.actual_digit || '-1';
        const actColor = pred.actual_color || '';
        const actLabel = pred.actual_label || '';
        const actBadge = (actLabel) ? buildPredBadge(actDigit, actColor, actLabel)
            : '<span class="pred-badge empty">--</span>';

        // Desktop row
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td class="period-cell">${period}</td>
            <td>${predBadge}</td>
            <td>${actBadge}</td>
            <td>${resultTag}</td>`;
        tbody.appendChild(tr);

        // Mobile card
        if (cards) {
            const card = document.createElement('div');
            card.className = `hist-card ${pred.correct === 1 ? 'hit' : pred.correct === 0 ? 'miss' : ''}`;
            card.innerHTML = `
                <div class="hist-card-period">#${period}</div>
                <div class="hist-card-body">
                    <div class="hist-card-col">
                        <span class="hist-card-label">Predicted</span>
                        ${predBadge}
                    </div>
                    <div class="hist-card-vs">vs</div>
                    <div class="hist-card-col">
                        <span class="hist-card-label">Actual</span>
                        ${actBadge}
                    </div>
                    <div class="hist-card-result">${resultTag}</div>
                </div>`;
            cards.appendChild(card);
        }
    });
}

// ============ INPUT HANDLERS ============
async function submitDigit(digit) {
    try {
        await fetch('/api/result', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ digit, source: 'manual', timer: activeTimer })
        });
    } catch (e) { addLog(`Error: ${e.message}`); }
}

async function submitBulk() {
    const input = $('#bulkInput');
    const digits = input.value.trim().split(/[\s,;]+/).map(Number).filter(d => d >= 0 && d <= 9);
    if (!digits.length) { addLog('Invalid bulk input.'); return; }
    addLog(`Submitting ${digits.length} results for ${activeTimer}...`);
    $('#trainingStatus').style.display = 'flex';
    try {
        const resp = await fetch('/api/results/bulk', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ digits, timer: activeTimer })
        });
        const data = await resp.json();
        input.value = '';
        addLog(`Bulk complete: ${data.count} results.`);
        $('#trainingStatus').style.display = 'none';
        fetchStats(); location.reload();
    } catch (e) { addLog(`Bulk error: ${e.message}`); $('#trainingStatus').style.display = 'none'; }
}

// ============ CONTROLS ============
async function triggerTrain() {
    addLog(`Training ${activeTimer} model...`);
    $('#trainingStatus').style.display = 'flex';
    try { await fetch(`/api/train?timer=${activeTimer}`, { method: 'POST' }); }
    catch (e) { addLog(`Training error: ${e.message}`); }
    $('#trainingStatus').style.display = 'none';
}

async function triggerPredict() {
    try {
        const resp = await fetch(`/api/predict?timer=${activeTimer}`);
        const data = await resp.json();
        if (data.error) { addLog(`Error: ${data.error}`); return; }
        data.timer = data.timer || activeTimer;
        timerPredictions[activeTimer] = data;
        renderPrediction(data);
    } catch (e) { addLog(`Predict error: ${e.message}`); }
}

async function resetModel() {
    if (!confirm(`Reset ${activeTimer} model?`)) return;
    try {
        await fetch(`/api/reset?timer=${activeTimer}`, { method: 'POST' });
        addLog(`[${activeTimer}] Model reset.`);
        timerPredictions[activeTimer] = null;
        clearAllPredictions();
    } catch (e) { addLog(`Reset error: ${e.message}`); }
}

async function toggleAutoPredict() {
    try {
        const resp = await fetch('/api/toggle-auto-predict', { method: 'POST' });
        const data = await resp.json();
        autoPredict = data.auto_predict;
        const btn = $('#btnAutoPredict');
        btn.textContent = `Auto-Predict: ${autoPredict ? 'ON' : 'OFF'}`;
        btn.classList.toggle('active', autoPredict);
        addLog(`Auto-predict: ${autoPredict ? 'ON' : 'OFF'}`);
    } catch (e) { addLog(`Toggle error: ${e.message}`); }
}

// ============ POLLER STATUS ============
async function checkPollerStatus() {
    try {
        const resp = await fetch('/api/poller/status');
        const data = await resp.json();
        const dot = $('.poller-dot');
        const text = $('.poller-text');
        if (data.active) {
            dot.className = 'poller-dot active';
            text.textContent = `API Live (${data.total || 0})`;
        } else {
            dot.className = 'poller-dot';
            text.textContent = 'API Offline';
        }
    } catch (e) { /* ignore */ }
}

// ============ LOGGING ============
function addLog(message) {
    const box = $('#logBox');
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    box.insertBefore(entry, box.firstChild);
    while (box.children.length > 50) box.removeChild(box.lastChild);
}

// ============ INIT ============
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    connectWebSocket();
    $$('.digit-btn').forEach(btn => btn.addEventListener('click', () => submitDigit(parseInt(btn.dataset.digit))));
    $('#bulkSubmit').addEventListener('click', submitBulk);
    $('#bulkInput').addEventListener('keypress', (e) => { if (e.key === 'Enter') submitBulk(); });
    $('#btnTrain').addEventListener('click', triggerTrain);
    $('#btnPredict').addEventListener('click', triggerPredict);
    $('#btnReset').addEventListener('click', resetModel);
    $('#btnAutoPredict').addEventListener('click', toggleAutoPredict);
    $$('.timer-tab').forEach(tab => tab.addEventListener('click', () => switchTimer(tab.dataset.timer)));
    fetchStats(); fetchPredictionHistory(); fetchFreshPrediction(activeTimer);
    checkPollerStatus();
    setInterval(checkPollerStatus, 30000);
});
