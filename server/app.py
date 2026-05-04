"""
FastAPI server — REST API + WebSocket for real-time updates + serves the dashboard.
Maintains separate models per timer (30sec, 1min, 3min).
Includes built-in API poller — no separate scraper needed.
"""
import os
import sys
import asyncio
import time
import json
import hashlib
import secrets
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Request, Cookie
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.database import init_db, add_result, get_recent_results, get_all_results_ordered, \
    get_accuracy_stats, add_prediction, update_prediction_result, get_recent_predictions, \
    get_result_count, get_timer_counts, extract_timer
from model.predictor import EnsemblePredictor
from config import SEQUENCE_LENGTH, HOST, PORT
from scraper.api_poller import WinGoPoller

# --- Global State ---
# Separate predictor per timer
TIMERS = ["30sec", "1min", "3min"]
predictors: dict[str, EnsemblePredictor] = {}
connected_clients: list[WebSocket] = []
latest_predictions: dict[str, dict] = {}  # per timer
pending_prediction_ids: dict[str, Optional[int]] = {}  # per timer
training_in_progress = False
auto_predict = True
poller_instance: Optional[WinGoPoller] = None
poller_status = {"active": False, "captured": {}}

# --- Auth ---
AUTH_USERNAME = os.getenv("DASHBOARD_USER", "Bibekjyoti")
AUTH_PASSWORD = os.getenv("DASHBOARD_PASS", "Bibek@7085")
SESSION_SECRET = os.getenv("SESSION_SECRET", secrets.token_hex(32))

def make_session_token(username: str) -> str:
    """Create a signed session token."""
    return hashlib.sha256(f"{SESSION_SECRET}:{username}".encode()).hexdigest()

VALID_TOKEN = make_session_token(AUTH_USERNAME)

# Paths that don't require auth
PUBLIC_PATHS = {"/login", "/api/login"}


# --- API Poller Callback ---
async def on_api_result(digit: int, round_id: str, color: str, timer: str):
    """Called by the API poller when a new result is detected."""
    # Store result
    result = await add_result(
        digit=digit,
        round_id=round_id,
        color=color,
        source=f"api_poller_{timer}",
        timer=timer,
    )

    # Update pending prediction for this timer
    if pending_prediction_ids.get(timer):
        correct = await update_prediction_result(
            pending_prediction_ids[timer], result["label"],
            actual_digit=digit, actual_color=color
        )
        result["prediction_correct"] = correct
        pending_prediction_ids[timer] = None

    # Online model update for this timer only
    results = await get_all_results_ordered(timer=timer)
    digits = [r["digit"] for r in results]
    timestamps = [r["timestamp"] for r in results]

    if timer in predictors:
        predictors[timer].online_update(digits, timestamps=timestamps)

    # Broadcast new result
    await broadcast({
        "type": "result",
        "data": result
    })

    # Auto-predict next for this timer
    if auto_predict and timer in predictors:
        await make_prediction_after_result(timer, digits, timestamps, round_id=round_id)

    # Get updated stats for this timer
    stats = await get_accuracy_stats(timer=timer)
    await broadcast({"type": "stats", "data": stats})

    # Update poller status
    poller_status["captured"][timer] = poller_status["captured"].get(timer, 0) + 1


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global poller_instance
    await init_db()

    # Initialize a separate model per timer
    for timer in TIMERS:
        predictors[timer] = EnsemblePredictor(timer_name=timer)
        predictors[timer].initialize_lstm()
        pending_prediction_ids[timer] = None
        latest_predictions[timer] = {}

        # Train on existing data for this timer
        results = await get_all_results_ordered(timer=timer)
        if len(results) >= SEQUENCE_LENGTH + 5:
            digits = [r["digit"] for r in results]
            timestamps = [r["timestamp"] for r in results]
            print(f"[STARTUP] Training {timer} model on {len(digits)} results...")
            predictors[timer].train_bulk(digits, epochs=50, lr=0.001, timestamps=timestamps)

    # Start background trainer
    asyncio.create_task(continuous_trainer())

    # Start API poller (replaces Selenium scraper)
    poller_instance = WinGoPoller(
        timers=TIMERS,
        on_new_result=on_api_result,
    )
    poller_status["active"] = True
    asyncio.create_task(poller_instance.run())
    print("[STARTUP] API poller started for all timers")

    yield

    # Shutdown
    if poller_instance:
        poller_instance.stop()
        poller_status["active"] = False

    for timer in TIMERS:
        predictors[timer].save()


app = FastAPI(title="Win Go Predictor", lifespan=lifespan)


# --- Auth Middleware ---
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # Allow public paths and static files
        if path in PUBLIC_PATHS or path.startswith("/static/") or path == "/ws":
            return await call_next(request)
        # Check session cookie
        session = request.cookies.get("session")
        if session != VALID_TOKEN:
            # API calls get 401, page requests get redirected
            if path.startswith("/api/"):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return RedirectResponse("/login")
        return await call_next(request)

app.add_middleware(AuthMiddleware)


# --- Pydantic Models ---
class ResultInput(BaseModel):
    digit: int
    round_id: str = None
    color: str = ""
    source: str = "manual"
    timer: str = None  # optional explicit timer


class BulkResultInput(BaseModel):
    digits: list[int]
    timer: str = "3min"


# --- WebSocket Manager ---
async def broadcast(message: dict):
    """Send message to all connected WebSocket clients."""
    dead = []
    for ws in connected_clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connected_clients.remove(ws)


# --- Background Tasks ---
async def continuous_trainer():
    """Periodically retrain each timer's model on its own data."""
    global training_in_progress
    while True:
        await asyncio.sleep(300)  # every 5 minutes
        try:
            for timer in TIMERS:
                count = await get_result_count(timer=timer)
                if count >= SEQUENCE_LENGTH + 10 and not training_in_progress:
                    training_in_progress = True
                    results = await get_all_results_ordered(timer=timer)
                    digits = [r["digit"] for r in results]
                    timestamps = [r["timestamp"] for r in results]
                    result = predictors[timer].train_bulk(digits, epochs=30, lr=0.0005, timestamps=timestamps)
                    print(f"[TRAINER] {timer} retrained: {result}")
                    training_in_progress = False
                    await broadcast({"type": "training_complete", "timer": timer, "result": result})
        except Exception as e:
            training_in_progress = False
            print(f"[TRAINER] Error: {e}")


async def make_prediction_after_result(timer: str, digits: list[int], timestamps: list[float] = None, round_id: str = None):
    """Generate and broadcast a prediction for a specific timer."""
    if len(digits) < 5:
        return

    # The prediction is for the NEXT round, so increment the round_id
    next_round_id = None
    if round_id:
        try:
            next_round_id = str(int(round_id) + 1)
        except (ValueError, TypeError):
            next_round_id = round_id  # fallback if non-numeric

    pred = predictors[timer].predict(digits, timestamps=timestamps)
    pred["timer"] = timer
    pred["for_round"] = next_round_id  # the round this prediction is FOR
    latest_predictions[timer] = pred

    # Store prediction with number and color
    num_pred = pred.get("number", {}).get("predicted", -1)
    color_pred = pred.get("color", {}).get("predicted", "")
    pending_prediction_ids[timer] = await add_prediction(
        predicted_label=pred["prediction"],
        confidence=pred["confidence"],
        timer=timer,
        predicted_number=num_pred,
        predicted_color=color_pred,
        round_id=next_round_id,
    )

    pred["pending_id"] = pending_prediction_ids[timer]

    await broadcast({
        "type": "prediction",
        "data": pred
    })


# --- API Endpoints ---

@app.post("/api/result")
async def submit_result(result_input: ResultInput):
    """Submit a single result."""
    if not 0 <= result_input.digit <= 9:
        return JSONResponse({"error": "Digit must be 0-9"}, status_code=400)

    # Determine timer
    timer = result_input.timer or extract_timer(result_input.source)

    # Store result
    result = await add_result(
        digit=result_input.digit,
        round_id=result_input.round_id,
        color=result_input.color,
        source=result_input.source,
        timer=timer
    )

    # Update pending prediction for this timer
    if pending_prediction_ids.get(timer):
        correct = await update_prediction_result(
            pending_prediction_ids[timer], result["label"],
            actual_digit=result_input.digit, actual_color=result_input.color
        )
        result["prediction_correct"] = correct
        pending_prediction_ids[timer] = None

    # Online model update for this timer only
    results = await get_all_results_ordered(timer=timer)
    digits = [r["digit"] for r in results]
    timestamps = [r["timestamp"] for r in results]

    if timer in predictors:
        predictors[timer].online_update(digits, timestamps=timestamps)

    # Broadcast new result
    await broadcast({
        "type": "result",
        "data": result
    })

    # Auto-predict next for this timer
    if auto_predict and timer in predictors:
        await make_prediction_after_result(timer, digits, timestamps)

    # Get updated stats for this timer
    stats = await get_accuracy_stats(timer=timer)
    await broadcast({"type": "stats", "data": stats})

    return {"status": "ok", "result": result}


@app.post("/api/results/bulk")
async def submit_bulk_results(bulk: BulkResultInput):
    """Submit multiple results at once for training."""
    timer = bulk.timer

    for digit in bulk.digits:
        if not 0 <= digit <= 9:
            return JSONResponse({"error": f"Invalid digit: {digit}"}, status_code=400)

    count = 0
    for digit in bulk.digits:
        await add_result(digit=digit, source="bulk", timer=timer)
        count += 1

    # Train this timer's model
    results = await get_all_results_ordered(timer=timer)
    digits = [r["digit"] for r in results]

    if timer in predictors:
        train_result = predictors[timer].train_bulk(digits, epochs=80, lr=0.001)
    else:
        train_result = {"status": "unknown_timer"}

    await broadcast({"type": "bulk_complete", "timer": timer, "count": count, "training": train_result})

    return {"status": "ok", "timer": timer, "count": count, "training": train_result}


@app.get("/api/predict")
async def get_prediction(timer: str = Query(default="3min")):
    """Get current prediction for a specific timer."""
    results = await get_all_results_ordered(timer=timer)
    digits = [r["digit"] for r in results]

    if len(digits) < 3:
        return {"error": "Need more data", "count": len(digits), "timer": timer}

    if timer in predictors:
        timestamps = [r["timestamp"] for r in results]
        pred = predictors[timer].predict(digits, timestamps=timestamps)
        pred["timer"] = timer
        return pred
    return {"error": f"Unknown timer: {timer}"}


@app.get("/api/stats")
async def get_stats(timer: str = Query(default=None)):
    """Get accuracy statistics, optionally per timer."""
    stats = await get_accuracy_stats(timer=timer)

    if timer and timer in predictors:
        model_status = predictors[timer].get_status()
    else:
        # Return combined status
        model_status = {t: predictors[t].get_status() for t in TIMERS if t in predictors}

    timer_counts = await get_timer_counts()

    return {**stats, "model": model_status, "timer_counts": timer_counts}


@app.get("/api/history")
async def get_history(limit: int = Query(default=100, le=500), timer: str = Query(default=None)):
    """Get recent results."""
    results = await get_recent_results(limit, timer=timer)
    return {"results": results, "timer": timer or "all"}


@app.get("/api/predictions")
async def get_predictions_history(limit: int = Query(default=50, le=200), timer: str = Query(default=None)):
    """Get recent predictions."""
    preds = await get_recent_predictions(limit, timer=timer)
    return {"predictions": preds, "timer": timer or "all"}


@app.post("/api/train")
async def trigger_training(timer: str = Query(default="3min")):
    """Manually trigger retraining for a specific timer."""
    global training_in_progress
    if training_in_progress:
        return {"status": "already_training"}

    results = await get_all_results_ordered(timer=timer)
    digits = [r["digit"] for r in results]

    if len(digits) < SEQUENCE_LENGTH + 5:
        return {"error": "Not enough data", "count": len(digits), "needed": SEQUENCE_LENGTH + 5, "timer": timer}

    if timer not in predictors:
        return {"error": f"Unknown timer: {timer}"}

    training_in_progress = True
    timestamps = [r["timestamp"] for r in results]
    result = predictors[timer].train_bulk(digits, epochs=100, lr=0.001, timestamps=timestamps)
    training_in_progress = False

    await broadcast({"type": "training_complete", "timer": timer, "result": result})
    return {**result, "timer": timer}


@app.post("/api/toggle-auto-predict")
async def toggle_auto():
    """Toggle auto-prediction after each result."""
    global auto_predict
    auto_predict = not auto_predict
    return {"auto_predict": auto_predict}


@app.post("/api/reset")
async def reset_model(timer: str = Query(default="3min")):
    """Reset a specific timer's model."""
    if timer in predictors:
        predictors[timer].initialize_lstm()
        return {"status": "model_reset", "timer": timer}
    return {"error": f"Unknown timer: {timer}"}


@app.get("/api/poller/status")
async def get_poller_status():
    """Get API poller status."""
    return {
        "active": poller_status["active"],
        "captured": poller_status["captured"],
        "total": sum(poller_status["captured"].values()),
        "timers": TIMERS,
    }


# --- WebSocket ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)

    try:
        # Send current state
        stats = await get_accuracy_stats()
        await websocket.send_json({"type": "stats", "data": stats})

        timer_counts = await get_timer_counts()
        await websocket.send_json({"type": "timer_counts", "data": timer_counts})

        recent = await get_recent_results(30)
        await websocket.send_json({"type": "history", "data": recent})

        # Send latest predictions for all timers
        for timer, pred in latest_predictions.items():
            if pred:
                await websocket.send_json({"type": "prediction", "data": pred})

        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg.get("type") == "switch_timer":
                # Client wants data for a specific timer
                req_timer = msg.get("timer", "3min")
                stats = await get_accuracy_stats(timer=req_timer)
                await websocket.send_json({"type": "stats", "data": stats})

                recent = await get_recent_results(30, timer=req_timer)
                await websocket.send_json({"type": "history", "data": recent})

                preds = await get_recent_predictions(30, timer=req_timer)
                await websocket.send_json({"type": "predictions_history", "data": preds})

                # Send cached prediction, or generate one on-the-fly
                if latest_predictions.get(req_timer):
                    await websocket.send_json({"type": "prediction", "data": latest_predictions[req_timer]})
                elif req_timer in predictors:
                    # No cached prediction — generate fresh
                    results = await get_all_results_ordered(timer=req_timer)
                    if len(results) >= 5:
                        digits = [r["digit"] for r in results]
                        pred = predictors[req_timer].predict(digits)
                        pred["timer"] = req_timer
                        latest_predictions[req_timer] = pred
                        await websocket.send_json({"type": "prediction", "data": pred})

    except WebSocketDisconnect:
        connected_clients.remove(websocket)
    except Exception:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


# --- Auth Endpoints ---

class LoginInput(BaseModel):
    username: str
    password: str


@app.post("/api/login")
async def login(creds: LoginInput):
    """Authenticate and set session cookie."""
    if creds.username == AUTH_USERNAME and creds.password == AUTH_PASSWORD:
        response = JSONResponse({"success": True})
        response.set_cookie(
            key="session",
            value=VALID_TOKEN,
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=30 * 24 * 3600,  # 30 days
        )
        return response
    return JSONResponse({"success": False, "error": "Invalid username or password"}, status_code=401)


@app.post("/api/logout")
async def logout():
    """Clear session cookie."""
    response = JSONResponse({"success": True})
    response.delete_cookie("session")
    return response


# --- Static Files (Dashboard) ---
dashboard_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard")


@app.get("/login")
async def serve_login():
    return FileResponse(os.path.join(dashboard_dir, "login.html"))


@app.get("/")
async def serve_dashboard():
    return FileResponse(os.path.join(dashboard_dir, "index.html"))


# Mount static files
if os.path.exists(dashboard_dir):
    app.mount("/static", StaticFiles(directory=dashboard_dir), name="static")
