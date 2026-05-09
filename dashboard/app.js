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
let trendChart = null;

let timerPredictions = { '30sec': null, '1min': null, '3min': null };
// Most recent prediction meta — used by paper trader to honor should_skip
let latestPredictionMeta = { should_skip: false, for_round: null, loss_streak: 0, timer: null };
// Golden hours — raw sorted list of {hour, max_loss, acc} from advanced_stats
let goldenHoursRaw = { week: [], month: [] };
// Whether the paper trader is restricted to golden hours only
let goldenHoursOnly = false;

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
            break;
        case 'timer_counts':
            updateTimerCounts(msg.data);
            break;
        case 'predictions_history':
            if (msg.timer === activeTimer || !msg.timer) {
                renderPredictionHistory(msg.data || []);
            }
            break;
        case 'advanced_stats':
            renderAdvancedStatsFromWS(msg.data);
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
    
    // Stop paper trader if they switch tabs to prevent cross-contamination of streaks
    if (paperTrader && paperTrader.active) {
        logPaperTrade(`[!] Timer switched to ${timer}. Stopping active trade to prevent data mixing.`, "#ef4444");
        stopPaperTrade();
    }
    
    activeTimer = timer;
    $$('.timer-tab').forEach(tab => tab.classList.toggle('active', tab.dataset.timer === timer));
    chartLabels.length = 0; chartAccData.length = 0; chartRecentData.length = 0;
    if (accuracyChart) accuracyChart.update('none');
    if (timerPredictions[timer]) renderPrediction(timerPredictions[timer]);
    else clearAllPredictions();
    
    // Send ONLY WebSocket message — server bundles stats, history, predictions, and advanced stats
    if (ws && ws.readyState === WebSocket.OPEN) {
        const offset = new Date().getTimezoneOffset() * -1;
        ws.send(JSON.stringify({ type: 'switch_timer', timer, tz_offset: offset }));
    }
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
    resultCount++;
    addLog(`Result: ${data.digit} (${data.label.toUpperCase()})${data.prediction_correct !== undefined ? (data.prediction_correct ? ' ✓' : ' ✗') : ''}`);
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

    // Confidence level badge
    const confBadge = $('#confidenceBadge');
    if (confBadge) {
        const skip    = data.should_skip === true;
        const flipped = data.flipped === true;
        if (flipped) {
            confBadge.textContent = '🔄 DIRECTION FLIPPED (stuck guard)';
            confBadge.className = 'confidence-badge low';
            confBadge.style.color = '#f59e0b';
        } else if (skip) {
            confBadge.textContent = '⚠ SKIP — LOW CONF / STREAK';
            confBadge.className = 'confidence-badge low';
            confBadge.style.color = '';
        } else {
            confBadge.textContent = '✓ OK TO BET';
            confBadge.className = 'confidence-badge high';
            confBadge.style.color = '';
        }
    }

    // Cache should_skip + active prediction round so paper trader can honor it
    latestPredictionMeta = {
        should_skip: data.should_skip === true,
        flipped: data.flipped === true,
        for_round: data.for_round,
        loss_streak: data.loss_streak || 0,
        timer: data.timer
    };

    // Loss streak indicator
    const streakEl = $('#lossStreakBadge');
    if (streakEl) {
        const streak = data.loss_streak || 0;
        if (streak >= 2) {
            streakEl.textContent = `🔥 ${streak} Loss Streak`;
            streakEl.style.display = 'inline-block';
        } else {
            streakEl.style.display = 'none';
        }
    }

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
    // NOTE: removed fetchPredictionHistory() call here — server sends it via WS bundle
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

// ============ ADVANCED STATS ============
async function fetchAdvancedStats(timer) {
    try {
        const offset = new Date().getTimezoneOffset(); // negative for UTC+ zones (e.g. IST = -330)
        const resp = await fetch(`/api/advanced_stats?timer=${timer}&tz_offset=${offset}`);
        const data = await resp.json();
        renderAdvancedStatsFromWS(data);
    } catch (e) {
        console.error('Advanced stats fetch failed:', e);
    }
}

// ============ GOLDEN HOURS FILTER ============
function updateGoldenHoursFilter() {
    goldenHoursOnly = $('#goldenHoursToggle').checked;

    // Update visual state of the custom toggle
    const slider = $('#goldenHoursSlider');
    const knob = $('#goldenHoursKnob');
    if (slider) slider.style.background = goldenHoursOnly ? 'rgba(245,158,11,0.6)' : '#334155';
    if (knob) {
        knob.style.background = goldenHoursOnly ? '#f59e0b' : '#94a3b8';
        knob.style.left = goldenHoursOnly ? '18px' : '2px';
    }

    const statusEl = $('#goldenHoursStatus');
    if (!statusEl) return;
    if (!goldenHoursOnly) {
        statusEl.style.display = 'none';
        return;
    }
    statusEl.style.display = 'block';
    const currentHour = new Date().getHours();
    refreshGoldenHoursStatus(currentHour);
}

function refreshGoldenHoursStatus(currentHour) {
    const statusEl = $('#goldenHoursStatus');
    if (!statusEl || !goldenHoursOnly) return;

    // Use month data (larger sample); fall back to week if month is empty
    const src = goldenHoursRaw.month.length >= 3 ? goldenHoursRaw.month : goldenHoursRaw.week;
    if (src.length === 0) {
        statusEl.textContent = 'Loading golden hours data...';
        statusEl.style.color = 'rgba(255,255,255,0.4)';
        return;
    }

    // Top 8 safest hours (lowest max_loss, then highest acc)
    const safeHours = src.slice(0, 8).map(h => h.hour);
    const inGoldenHour = safeHours.includes(currentHour);

    const formatHour = h => {
        const ampm = h >= 12 ? 'PM' : 'AM';
        const h12 = h % 12 || 12;
        return `${h12}${ampm}`;
    };

    const safeLabels = src.slice(0, 8)
        .map(h => `${formatHour(h.hour)}(L:${h.max_loss})`)
        .join(' · ');

    if (inGoldenHour) {
        statusEl.innerHTML = `<span style="color:#10b981;">&#10003; ${formatHour(currentHour)}-${formatHour(currentHour+1)} is a golden hour</span><br>${safeLabels}`;
    } else {
        statusEl.innerHTML = `<span style="color:#ef4444;">&#9888; ${formatHour(currentHour)}-${formatHour(currentHour+1)} is NOT a golden hour — bets will be skipped</span><br>${safeLabels}`;
    }
}

function renderAdvancedStatsFromWS(data) {
    if (!data) return;

    // Store raw golden hours for paper trader gating
    if (data.safest_hours_raw) {
        goldenHoursRaw.week  = data.safest_hours_raw.week  || [];
        goldenHoursRaw.month = data.safest_hours_raw.month || [];
        // Refresh the status display if filter is active
        if (goldenHoursOnly) refreshGoldenHoursStatus(new Date().getHours());
    }

    // Render Streaks
    if (data.streaks) {
        $('#streakWinToday').textContent = data.streaks.today.max_wins;
        $('#streakLossToday').textContent = data.streaks.today.max_losses;
        $('#streakWinWeek').textContent = data.streaks.week.max_wins;
        $('#streakLossWeek').textContent = data.streaks.week.max_losses;
        $('#streakWinMonth').textContent = data.streaks.month.max_wins;
        $('#streakLossMonth').textContent = data.streaks.month.max_losses;
    }

    // Render Golden Times
    const oToday = data.optimal_times?.today || 'No data yet today';
    const oWeek = data.optimal_times?.week || 'No optimal hours detected';
    const oMonth = data.optimal_times?.month || 'No optimal hours detected';
    $('#optimalTimesToday').innerHTML = oToday;
    $('#optimalTimesWeek').innerHTML = oWeek;
    $('#optimalTimesMonth').innerHTML = oMonth;

    // Render Trend Chart
    if (data.trend) {
        updateTrendChart(data.trend);
    }
}

function updateTrendChart(trendData) {
    const labels = trendData.map(d => d.date);
    const accData = trendData.map(d => d.accuracy);

    if (!trendChart) {
        const ctx = document.getElementById('trendChart').getContext('2d');
        trendChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Daily Accuracy (%)',
                    data: accData,
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    borderWidth: 2,
                    pointBackgroundColor: '#8b5cf6',
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: { mode: 'index', intersect: false }
                },
                scales: {
                    y: { 
                        beginAtZero: true, 
                        max: 100,
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono' } }
                    },
                    x: { 
                        grid: { display: false },
                        ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono' } }
                    }
                }
            }
        });
    } else {
        trendChart.data.labels = labels;
        trendChart.data.datasets[0].data = accData;
        trendChart.update();
    }
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
    // Pass to strategy calculator
    trackLiveStreakFromHistory(predictions);

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

// ============ RISK MANAGEMENT STRATEGY ============
let strategyMode = 'capital';
let currentActiveLossStreak = 0;

function setStrategyMode(mode) {
    strategyMode = mode;
    const btnCapital = $('#modeCapital');
    const btnBet = $('#modeBet');
    
    if (mode === 'capital') {
        btnCapital.classList.add('active');
        btnBet.classList.remove('active');
        $('#primaryInputLabel').textContent = 'Total Capital (₹)';
        $('#strategyInput').value = '1000';
    } else {
        btnBet.classList.add('active');
        btnCapital.classList.remove('active');
        $('#primaryInputLabel').textContent = 'Initial Base Bet (₹)';
        $('#strategyInput').value = '10';
    }
    calculateStrategy();
}

function calculateMartingaleProgression(baseBet, levels, customMultiplier, payoutMultiplier = 1.96) {
    let progression = [];
    let totalLoss = 0;
    let currentBet = baseBet;
    
    for (let i = 1; i <= levels; i++) {
        if (i === 1) {
            currentBet = baseBet;
        } else {
            // Use user-defined multiplier, e.g. 2.5x, 3x
            currentBet = progression[progression.length - 1].bet * customMultiplier;
            currentBet = Math.ceil(currentBet); // Round up to nearest integer
        }
        
        const profitIfWin = (currentBet * payoutMultiplier) - (totalLoss + currentBet);
        const totalRisk = totalLoss + currentBet;
        
        progression.push({
            level: i,
            bet: currentBet,
            totalRisk: totalRisk,
            profit: profitIfWin
        });
        
        totalLoss += currentBet;
    }
    return progression;
}

function findOptimalBaseBet(totalCapital, levels, customMultiplier) {
    // Binary search to find the highest base bet that keeps totalRisk <= totalCapital
    let low = 0.01;
    let high = totalCapital;
    let bestBet = 0;
    let maxIterations = 100;
    
    for (let i = 0; i < maxIterations; i++) {
        let mid = (low + high) / 2;
        let prog = calculateMartingaleProgression(mid, levels, customMultiplier);
        let maxRisk = prog[prog.length - 1].totalRisk;
        
        if (maxRisk <= totalCapital) {
            bestBet = mid;
            low = mid; // Try going higher
        } else {
            high = mid; // Too expensive, go lower
        }
        
        if (high - low < 0.01) break;
    }
    
    // Round down to nearest integer for safety
    return Math.floor(bestBet);
}

function calculateStrategy() {
    const inputVal = parseFloat($('#strategyInput').value);
    const levels = parseInt($('#strategyLevels').value);
    let customMult = parseFloat($('#strategyMultiplier').value);
    
    if (isNaN(inputVal) || inputVal <= 0 || isNaN(levels) || levels <= 0) return;
    if (isNaN(customMult) || customMult < 1.1) customMult = 2.08;
    
    let baseBet = 0;
    let totalCap = 0;
    let progression = [];
    
    if (strategyMode === 'capital') {
        baseBet = findOptimalBaseBet(inputVal, levels, customMult);
        progression = calculateMartingaleProgression(baseBet, levels, customMult);
        totalCap = progression.length > 0 ? progression[progression.length - 1].totalRisk : 0;
    } else {
        baseBet = inputVal;
        progression = calculateMartingaleProgression(baseBet, levels, customMult);
        totalCap = progression.length > 0 ? progression[progression.length - 1].totalRisk : 0;
    }
    
    // Update summary UI
    $('#calcBaseBet').textContent = `₹${baseBet.toLocaleString()}`;
    $('#calcTotalCap').textContent = `₹${totalCap.toLocaleString()}`;
    
    // Render Table
    let tableHtml = '';
    let hasLosses = false;
    
    progression.forEach(p => {
        if (p.profit < 0) hasLosses = true;
        
        let profitColor = p.profit >= 0 ? "#10b981" : "#ef4444";
        let profitSign = p.profit >= 0 ? "+" : "";
        
        tableHtml += `
            <tr>
                <td>Level ${p.level}</td>
                <td style="color: #60a5fa">₹${p.bet.toLocaleString()}</td>
                <td style="color: #ef4444">₹${p.totalRisk.toLocaleString()}</td>
                <td style="color: ${profitColor}">${profitSign}₹${p.profit.toFixed(2)}</td>
            </tr>
        `;
    });
    
    // Warn if capital mode can't even afford 1 base bet
    if (strategyMode === 'capital' && baseBet < 1) {
        $('#calcBaseBet').textContent = "Not Enough Capital";
        $('#calcBaseBet').style.color = "#ef4444";
        $('#progressionTableBody').innerHTML = `<tr><td colspan="4" style="text-align:center;color:#ef4444">Increase capital or decrease levels to afford at least a ₹1 base bet.</td></tr>`;
        return;
    } else if (hasLosses) {
        $('#calcBaseBet').style.color = "#f59e0b"; // warning color
        $('#progressionTableBody').innerHTML = `<tr><td colspan="4" style="text-align:center;background:rgba(239, 68, 68, 0.1);color:#ef4444;font-weight:bold;padding:10px;">WARNING: Multiplier is too low! You will lose money even if you win at higher levels. (Requires at least 2.05x)</td></tr>` + tableHtml;
    } else {
        $('#calcBaseBet').style.color = "#10b981";
        $('#progressionTableBody').innerHTML = tableHtml;
    }
}

function updateLiveAlert() {
    const waitStreak = parseInt($('#strategyWait').value) || 0;
    const levels = parseInt($('#strategyLevels').value) || 7;
    const box = $('#liveAlertBox');
    const status = $('#alertStatus');
    const msg = $('#alertMessage');
    
    // Icons
    const iconWait = `<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>`;
    const iconEnter = `<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`;
    const iconAlert = `<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
    
    // We need the current progression array to fetch bet amounts
    // (calculateStrategy() builds this table, but we can quickly rebuild it here to get the exact amount)
    let customMult = parseFloat($('#strategyMultiplier').value);
    if (isNaN(customMult) || customMult < 1.1) customMult = 2.08;
    const baseBetText = $('#calcBaseBet').textContent.replace(/[^0-9.-]+/g, "");
    const baseBet = parseFloat(baseBetText) || 10;
    const prog = calculateMartingaleProgression(baseBet, levels, customMult);
    
    if (currentActiveLossStreak >= waitStreak && waitStreak >= 0) {
        // We are IN the market
        const currentLevelIndex = currentActiveLossStreak - waitStreak; // 0-indexed (0 = Level 1)
        const currentLevel = currentLevelIndex + 1;
        
        if (currentLevel <= levels) {
            // Active sequence! Tell them exactly what to bet
            const exactBet = prog[currentLevelIndex] ? prog[currentLevelIndex].bet : 0;
            box.className = 'alert-box enter';
            status.textContent = `ACTIVE: PLACE LEVEL ${currentLevel} BET`;
            msg.textContent = `AI Loss Streak: ${currentActiveLossStreak}. You are on Level ${currentLevel} of your strategy. Place a bet of ₹${exactBet.toLocaleString()} now.`;
            $('#alertIcon').innerHTML = iconEnter;
        } else {
            // Oh no, the streak exceeded their max levels!
            box.className = 'alert-box waiting';
            box.style.border = "1px solid #ef4444";
            box.style.background = "rgba(239, 68, 68, 0.1)";
            status.textContent = 'MAX LEVELS EXCEEDED';
            status.style.color = '#ef4444';
            msg.textContent = `AI Loss Streak: ${currentActiveLossStreak}. You have exceeded your ${levels} safe levels. Wait for a win to reset.`;
            $('#alertIcon').innerHTML = iconAlert;
            $('#alertIcon').style.color = "#ef4444";
        }
    } else {
        // We are WAITING
        box.className = 'alert-box waiting';
        box.style.border = "";
        box.style.background = "";
        status.textContent = 'WAITING...';
        status.style.color = '';
        msg.textContent = `AI is currently on a ${currentActiveLossStreak}-loss streak. Wait for ${waitStreak} before entering.`;
        $('#alertIcon').innerHTML = iconWait;
        $('#alertIcon').style.color = '';
    }
}

// Track Live Streaks from websocket updates
function trackLiveStreakFromHistory(historyArray) {
    if (!historyArray || historyArray.length === 0) return;
    
    let streak = 0;
    for (let i = 0; i < historyArray.length; i++) {
        // Skip pending rounds that haven't resulted yet
        if (historyArray[i].correct === null || historyArray[i].correct === undefined) {
            continue;
        }
        if (historyArray[i].correct === 0) {
            streak++;
        } else {
            break;
        }
    }
    currentActiveLossStreak = streak;
    updateLiveAlert();
    processPaperTrade(historyArray);
}

// ============ STRATEGY PERSISTENCE ============
function saveStrategySettings() {
    const settings = {
        mode: strategyMode,
        input: $('#strategyInput').value,
        levels: $('#strategyLevels').value,
        multiplier: $('#strategyMultiplier').value,
        wait: $('#strategyWait').value
    };
    localStorage.setItem('riskStrategy', JSON.stringify(settings));
    
    const toast = $('#saveToast');
    toast.style.opacity = 1;
    setTimeout(() => { toast.style.opacity = 0; }, 2000);
}

function loadStrategySettings() {
    try {
        const saved = localStorage.getItem('riskStrategy');
        if (saved) {
            const settings = JSON.parse(saved);
            $('#strategyInput').value = settings.input;
            $('#strategyLevels').value = settings.levels;
            $('#strategyMultiplier').value = settings.multiplier || 2.08;
            $('#strategyWait').value = settings.wait;
            setStrategyMode(settings.mode || 'capital');
        }
    } catch (e) {
        console.error("Failed to load saved strategy");
    }
}

// ============ PAPER TRADING ENGINE ============
let paperTrader = {
    active: false,
    balance: 0,
    lastProcessedRound: null,
    currentBetLevel: -1, // -1 means waiting
    pendingBet: false,   // true only when a bet was actually placed last round
    waitStreakTarget: 0,
    maxLevels: 0,
    multiplier: 2.08,
    baseBet: 10,
    progression: []
};

function togglePaperTrade() {
    if (paperTrader.active) return;
    
    const capital = parseFloat($('#strategyInput').value);
    if (isNaN(capital) || capital <= 0) {
        alert("Please set a valid Total Capital before starting.");
        return;
    }
    
    paperTrader.active = true;
    paperTrader.balance = capital;
    paperTrader.waitStreakTarget = parseInt($('#strategyWait').value) || 0;
    paperTrader.maxLevels = parseInt($('#strategyLevels').value) || 7;
    paperTrader.multiplier = parseFloat($('#strategyMultiplier').value) || 2.08;
    
    const baseBetText = $('#calcBaseBet').textContent.replace(/[^0-9.-]+/g, "");
    paperTrader.baseBet = parseFloat(baseBetText) || 10;
    
    if (paperTrader.baseBet < 1) {
        alert("Base bet must be at least ₹1. Adjust your settings.");
        paperTrader.active = false;
        return;
    }
    
    paperTrader.progression = calculateMartingaleProgression(paperTrader.baseBet, paperTrader.maxLevels, paperTrader.multiplier);
    paperTrader.currentBetLevel = -1; // Waiting
    paperTrader.pendingBet = false;
    paperTrader.lastProcessedRound = null;
    
    $('#paperStatusBadge').textContent = 'RUNNING';
    $('#paperStatusBadge').style.background = 'rgba(16, 185, 129, 0.2)';
    $('#paperStatusBadge').style.color = '#10b981';
    
    $('#paperLogBox').innerHTML = `<div style="color: #10b981; margin-bottom: 5px;">[SYSTEM] Engine Started. Bankroll: ₹${capital.toLocaleString()}</div>`;
    updatePaperUI();
}

function stopPaperTrade() {
    if (!paperTrader.active) return;
    paperTrader.active = false;
    $('#paperStatusBadge').textContent = 'STOPPED';
    $('#paperStatusBadge').style.background = 'rgba(239, 68, 68, 0.2)';
    $('#paperStatusBadge').style.color = '#ef4444';
    logPaperTrade(`[SYSTEM] Engine Stopped. Final Bankroll: ₹${paperTrader.balance.toLocaleString()}`, "#ef4444");
}

function updatePaperUI() {
    $('#paperBalance').textContent = `₹${paperTrader.balance.toFixed(2)}`;
}

function logPaperTrade(msg, color="#94a3b8") {
    const box = $('#paperLogBox');
    const div = document.createElement('div');
    div.style.color = color;
    div.style.marginBottom = '4px';
    div.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
}

function processPaperTrade(historyArray) {
    if (!paperTrader.active) return;
    
    const resolved = historyArray.filter(h => h.correct !== null && h.correct !== undefined);
    if (resolved.length === 0) return;
    const latest = resolved[0];
    
    if (latest.round_id === paperTrader.lastProcessedRound) return; // No new data
    
    // ── 1. Evaluate PREVIOUS round's result (only if we actually placed a bet) ──
    if (paperTrader.currentBetLevel !== -1 && paperTrader.pendingBet) {
        const betAmount = paperTrader.progression[paperTrader.currentBetLevel].bet;
        
        if (latest.correct === 1) {
            const payout = betAmount * 1.96;
            paperTrader.balance += payout;
            logPaperTrade(`WON Level ${paperTrader.currentBetLevel + 1}! (+₹${payout.toFixed(2)})`, "#10b981");
            paperTrader.currentBetLevel = -1;
        } else {
            logPaperTrade(`LOST Level ${paperTrader.currentBetLevel + 1}.`, "#ef4444");
            paperTrader.currentBetLevel++;
            if (paperTrader.currentBetLevel >= paperTrader.maxLevels) {
                logPaperTrade(`CRITICAL: Max Levels Exceeded. Sequence Wiped.`, "#ef4444");
                paperTrader.currentBetLevel = -1;
            }
        }
        paperTrader.pendingBet = false;
    }
    
    // ── 2. Calculate current AI loss streak ──────────────────────────────────────
    let newStreak = 0;
    for (let i = 0; i < resolved.length; i++) {
        if (resolved[i].correct === 0) newStreak++;
        else break;
    }
    
    // ── 3. Determine entry level ──────────────────────────────────────────────────
    if (paperTrader.currentBetLevel === -1) {
        if (newStreak >= paperTrader.waitStreakTarget && paperTrader.waitStreakTarget >= 0) {
            paperTrader.currentBetLevel = newStreak - paperTrader.waitStreakTarget; 
            if (paperTrader.currentBetLevel >= paperTrader.maxLevels) {
                paperTrader.currentBetLevel = -1;
            }
        }
    }
    
    // ── 4. Decide whether to place a bet for the NEXT round ──────────────────────
    if (paperTrader.currentBetLevel !== -1) {

        // ── AI skip gate (server-side: streak / cold-regime / golden-hours) ────
        const meta = latestPredictionMeta;
        if (meta && meta.timer === activeTimer && meta.should_skip) {
            const fmt = h => `${h % 12 || 12}${h >= 12 ? 'PM' : 'AM'}`;
            let reason;
            if (meta.golden_hour_skip) {
                const h = new Date().getHours();
                reason = `${fmt(h)} not a golden hour`;
            } else if (meta.cold_regime) {
                const acc = meta.rolling_acc != null ? ` (${(meta.rolling_acc*100).toFixed(0)}% last 10)` : '';
                reason = `cold streak${acc}`;
            } else {
                reason = `loss streak (${meta.loss_streak ?? newStreak})`;
            }
            logPaperTrade(`⏸ PAUSED L${paperTrader.currentBetLevel + 1}: AI skip — ${reason}`, "#f59e0b");
            paperTrader.pendingBet = false;
            paperTrader.lastProcessedRound = latest.round_id;
            updatePaperUI();
            return;
        }

        // ── Client-side golden hours filter (toggle in UI) ────────────────────
        if (goldenHoursOnly) {
            const currentHour = new Date().getHours();
            const src = goldenHoursRaw.month.length >= 3
                ? goldenHoursRaw.month : goldenHoursRaw.week;
            const safeHours = src.slice(0, 8).map(h => h.hour);
            if (safeHours.length > 0 && !safeHours.includes(currentHour)) {
                const fmt = h => `${h % 12 || 12}${h >= 12 ? 'PM' : 'AM'}`;
                logPaperTrade(
                    `⏸ PAUSED (${fmt(currentHour)} not a golden hour — UI filter)`,
                    "#f59e0b"
                );
                paperTrader.pendingBet = false;
                paperTrader.lastProcessedRound = latest.round_id;
                updatePaperUI();
                return;
            }
        }

        // ── Place the bet ─────────────────────────────────────────────────────
        const nextBet = paperTrader.progression[paperTrader.currentBetLevel].bet;
        if (paperTrader.balance < nextBet) {
            logPaperTrade(`BANKRUPT! Insufficient funds for Level ${paperTrader.currentBetLevel + 1} (₹${nextBet}).`, "#ef4444");
            stopPaperTrade();
            return;
        }
        paperTrader.balance -= nextBet;
        logPaperTrade(`PLACED Level ${paperTrader.currentBetLevel + 1} Bet: -\u20b9${nextBet.toLocaleString()}`, "#60a5fa");
        paperTrader.pendingBet = true;
    }
    
    paperTrader.lastProcessedRound = latest.round_id;
    updatePaperUI();
}

// ============ INIT ============
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    connectWebSocket();
    $('#btnTrain').addEventListener('click', triggerTrain);
    $('#btnPredict').addEventListener('click', triggerPredict);
    $('#btnReset').addEventListener('click', resetModel);
    $('#btnAutoPredict').addEventListener('click', toggleAutoPredict);
    $$('.timer-tab').forEach(tab => tab.addEventListener('click', () => switchTimer(tab.dataset.timer)));
    fetchStats(); fetchPredictionHistory(); fetchFreshPrediction(activeTimer); fetchAdvancedStats(activeTimer);
    checkPollerStatus();
    setInterval(checkPollerStatus, 30000);
    setInterval(() => fetchAdvancedStats(activeTimer), 60000); // refresh advanced stats every minute
    setInterval(() => refreshGoldenHoursStatus(new Date().getHours()), 60000); // re-check hour
    
    // Initialize Strategy Engine
    loadStrategySettings();
    calculateStrategy();
    updateLiveAlert();
});
