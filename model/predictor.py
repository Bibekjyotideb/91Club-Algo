"""
LSTM-based prediction model + ensemble predictor.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from collections import deque

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEQUENCE_LENGTH, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, MODEL_DIR
from model.features import FeatureExtractor, MarkovChain, PatternDetector, FrequencyAnalyzer


class LSTMPredictor(nn.Module):
    """LSTM network for binary sequence prediction."""

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use last timestep output
        last_output = lstm_out[:, -1, :]
        return self.classifier(last_output).squeeze(-1)


class EnsemblePredictor:
    """
    Combines LSTM, Markov Chain, Pattern Detector, and Frequency Analyzer
    for robust predictions with weighted voting.
    """

    def __init__(self, timer_name: str = "3min"):
        self.timer_name = timer_name
        self.feature_extractor = FeatureExtractor()
        self.markov = MarkovChain(order=4)
        self.pattern_detector = PatternDetector()
        self.freq_analyzer = FrequencyAnalyzer()

        # LSTM model
        self.lstm_model: Optional[LSTMPredictor] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensemble weights (auto-adjusted based on performance)
        self.weights = {
            "lstm": 0.45,
            "markov": 0.25,
            "pattern": 0.15,
            "frequency": 0.15
        }

        # Performance tracking for weight adjustment
        self.model_correct = {"lstm": 0, "markov": 0, "pattern": 0, "frequency": 0}
        self.model_total = {"lstm": 0, "markov": 0, "pattern": 0, "frequency": 0}

        # Streak tracking and live weight adjustment
        self.recent_outcomes = deque(maxlen=30)  # True/False for ensemble correctness
        self._pending_prediction = None  # stores per-model predictions for outcome tracking

        # Track recently predicted digits to enforce variety in number predictions.
        self._last_digit_predictions: deque = deque(maxlen=8)

        self._initialized = False

    def initialize_lstm(self):
        """Initialize or load the LSTM model."""
        input_size = self.feature_extractor.num_features
        self.lstm_model = LSTMPredictor(input_size=input_size).to(self.device)

        # Try to load saved model (per-timer checkpoint)
        model_path = os.path.join(MODEL_DIR, f"best_lstm_{self.timer_name}.pt")
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                self.lstm_model.load_state_dict(checkpoint["model_state"])
                self.weights = checkpoint.get("ensemble_weights", self.weights)
                self.model_correct = checkpoint.get("model_correct", self.model_correct)
                self.model_total = checkpoint.get("model_total", self.model_total)
                # Restore streak history if available
                saved_outcomes = checkpoint.get("recent_outcomes", [])
                self.recent_outcomes = deque(saved_outcomes, maxlen=30)
                # Restore digit diversity history
                saved_digits = checkpoint.get("last_digit_predictions", [])
                self._last_digit_predictions = deque(saved_digits, maxlen=8)
                print(f"[MODEL] [{self.timer_name}] Loaded saved model from {model_path}")
            except Exception as e:
                print(f"[MODEL] [{self.timer_name}] Could not load saved model: {e}")

        self._initialized = True

    def train_bulk(self, digits: list[int], epochs: int = 100, lr: float = 0.001,
                   timestamps: list[float] = None) -> dict:
        """Train the LSTM on bulk historical data."""
        if not self._initialized:
            self.initialize_lstm()

        # Train Markov chain
        labels = [1 if d >= 5 else 0 for d in digits]
        self.markov.train(labels)

        # Prepare LSTM data (gap-aware if timestamps available)
        X, y = self.feature_extractor.extract_sequence_features(digits, SEQUENCE_LENGTH, timestamps)

        if len(X) < 10:
            return {"status": "insufficient_data", "samples": len(X)}

        # Train/val split
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        best_val_acc = 0
        best_state = None
        train_losses = []

        self.lstm_model.train()
        for epoch in range(epochs):
            # Mini-batch training
            indices = torch.randperm(len(X_train_t))
            batch_size = min(32, len(X_train_t))
            epoch_loss = 0
            num_batches = 0

            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start + batch_size]
                batch_X = X_train_t[batch_idx]
                batch_y = y_train_t[batch_idx]

                optimizer.zero_grad()
                output = self.lstm_model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lstm_model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            train_losses.append(avg_loss)

            # Validation
            if (epoch + 1) % 10 == 0:
                self.lstm_model.eval()
                with torch.no_grad():
                    val_output = self.lstm_model(X_val_t)
                    val_preds = (val_output > 0.5).float()
                    val_acc = (val_preds == y_val_t).float().mean().item()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = self.lstm_model.state_dict().copy()

                self.lstm_model.train()
                print(f"[TRAIN] [{self.timer_name}] Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.1%}")

        # Restore best model
        if best_state:
            self.lstm_model.load_state_dict(best_state)

        # Save
        self._save_model()

        return {
            "status": "trained",
            "epochs": epochs,
            "samples": len(X),
            "best_val_accuracy": round(best_val_acc * 100, 1),
            "final_loss": train_losses[-1] if train_losses else 0
        }

    def online_update(self, digits: list[int], lr: float = 0.0005,
                      timestamps: list[float] = None):
        """Update the model with the latest data point (online learning).
        Skips LSTM updates during loss streaks to prevent noise-chasing."""
        if not self._initialized or len(digits) < SEQUENCE_LENGTH + 1:
            return

        # Update Markov chain (always — it's stateless and cheap)
        labels = [1 if d >= 5 else 0 for d in digits]
        self.markov.train(labels)

        # Freeze LSTM during 3+ loss streaks to prevent chasing noise
        if self._get_loss_streak() >= 3:
            return

        # Quick LSTM update on recent data
        n = SEQUENCE_LENGTH + 10
        recent = digits[-n:]
        recent_ts = timestamps[-n:] if timestamps else None
        X, y = self.feature_extractor.extract_sequence_features(recent, SEQUENCE_LENGTH, recent_ts)

        if len(X) < 1:
            return

        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.FloatTensor(y).to(self.device)

        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        self.lstm_model.train()
        for _ in range(3):  # few quick updates
            optimizer.zero_grad()
            output = self.lstm_model(X_t)
            loss = criterion(output, y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.lstm_model.parameters(), 1.0)
            optimizer.step()

    def predict(self, digits: list[int], timestamps: list[float] = None) -> dict:
        """
        Generate ensemble prediction for the next outcome.

        Returns:
            {
                "prediction": "small" or "big",
                "confidence": float (0-1),
                "prob_big": float (0-1),
                "models": {model_name: {prob_big, confidence}},
                "number": {"predicted": int, "confidence": float, "probabilities": list[float]},
                "color": {"predicted": str, "confidence": float, "probabilities": {"red": float, "green": float, "violet": float}},
            }
        """
        if not self._initialized:
            self.initialize_lstm()

        labels = [1 if d >= 5 else 0 for d in digits]
        model_predictions = {}

        # 1. LSTM prediction
        lstm_prob = 0.5
        lstm_conf = 0.0
        if len(digits) >= SEQUENCE_LENGTH:
            try:
                self.lstm_model.eval()
                with torch.no_grad():
                    X = self.feature_extractor.extract_latest_sequence(
                        digits, SEQUENCE_LENGTH, timestamps
                    )
                    X_t = torch.FloatTensor(X).to(self.device)
                    output = self.lstm_model(X_t)
                    lstm_prob = output.item()
                    lstm_conf = abs(lstm_prob - 0.5) * 2  # confidence = distance from 0.5
            except Exception as e:
                print(f"[MODEL] LSTM prediction error: {e}")
        model_predictions["lstm"] = {"prob_big": lstm_prob, "confidence": lstm_conf}

        # 2. Markov chain
        markov_prob, markov_conf = self.markov.predict(labels)
        model_predictions["markov"] = {"prob_big": markov_prob, "confidence": markov_conf}

        # 3. Pattern detector
        pattern_result = self.pattern_detector.detect_patterns(labels)
        model_predictions["pattern"] = {
            "prob_big": pattern_result["prob_big"],
            "confidence": pattern_result["confidence"]
        }

        # 4. Frequency analyzer
        freq_result = self.freq_analyzer.analyze(digits)
        model_predictions["frequency"] = {
            "prob_big": freq_result["prob_big"],
            "confidence": freq_result["confidence"]
        }

        # Weighted ensemble
        weighted_prob = 0
        total_weight = 0
        for model_name, pred in model_predictions.items():
            w = self.weights[model_name] * (0.5 + pred["confidence"] * 0.5)
            weighted_prob += pred["prob_big"] * w
            total_weight += w

        if total_weight > 0:
            ensemble_prob = weighted_prob / total_weight
        else:
            ensemble_prob = 0.5

        # Loss streak (used for skip / confidence gating below)
        loss_streak = self._get_loss_streak()

        # --- Streak-aware confidence reduction ---
        # During a loss streak, nudge ensemble_prob toward 0.5 (never past it).
        # This DOES NOT flip the prediction direction — it only lowers confidence
        # so the HIGH-confidence gate skips those rounds, protecting bankroll.
        if loss_streak >= 3:
            contrarian_strength = min(0.10 + (loss_streak - 3) * 0.06, 0.22)
            if ensemble_prob > 0.5:
                ensemble_prob = max(ensemble_prob - contrarian_strength, 0.50)
            else:
                ensemble_prob = min(ensemble_prob + contrarian_strength, 0.50)

        # Store pending prediction for outcome tracking
        self._pending_prediction = {
            "ensemble_big": ensemble_prob >= 0.5,
            "models": {name: pred["prob_big"] >= 0.5 for name, pred in model_predictions.items()}
        }

        # ===== SIZE PREDICTION — directly from ensemble (primary, most reliable) =====
        # Number and color are derived FROM the size decision, never the reverse.
        # Deriving size from predicted_digit was a bug: the diversity penalty inside
        # _predict_number could push argmax to the wrong side, silently inverting size.
        size_pred = "big" if ensemble_prob >= 0.5 else "small"
        size_conf = abs(ensemble_prob - 0.5) * 2

        # ===== NUMBER PREDICTION (0-9) — informational, constrained to correct side =====
        number_pred = self._predict_number(digits, ensemble_prob_big=ensemble_prob)
        predicted_digit = number_pred["predicted"]
        probs = number_pred["probabilities"]  # 10 probabilities
        prob_big_from_num = float(sum(probs[5:10]))  # used for probability display bars

        # ===== CONFIDENCE GATING =====
        # Backtests on 8900+ rounds confirm all sub-models sit within ±2pp of 50%
        # → ensemble-distance and model-agreement are pure noise on random data.
        # Using them to gate "HIGH" confidence was actively harmful (backtest showed
        # HIGH-conf rounds at 45.8% — worse than flipping a coin).
        #
        # Real risk control: only skip on loss streaks ≥ 4 (bankroll protection).
        # Model-agreement is still computed for informational display bars.
        ensemble_big = ensemble_prob >= 0.5
        agree_count = sum(
            1 for p in model_predictions.values()
            if (p["prob_big"] >= 0.5) == ensemble_big
        )
        # Always "HIGH" unless on a damaging loss streak
        confidence_level = "HIGH" if loss_streak < 4 else "LOW"
        should_skip = loss_streak >= 4

        # ===== DERIVE COLOR deterministically from the predicted number =====
        # Even (incl 0) = Red, Odd = Green, 0 & 5 = also Violet
        color_pred = self._predict_color(probs, predicted_digit)

        return {
            "prediction": size_pred,
            "confidence": round(size_conf, 4),
            "prob_big": round(prob_big_from_num, 4),
            "models": model_predictions,
            "number": number_pred,
            "color": color_pred,
            "confidence_level": confidence_level,
            "loss_streak": loss_streak,
            "should_skip": should_skip,
            "flipped": False,
            "model_agreement": agree_count,
        }

    def _predict_number(self, digits: list[int], ensemble_prob_big: float = 0.5) -> dict:
        """
        Predict the most likely next digit (0-9).
        
        This is the CORE prediction — color and size are derived from this.
        
        Uses:
          1. Frequency analysis — which digits appear most/least
          2. Digit-level Markov chain — transition probabilities between digits
          3. Recency weighting — recent digits weighted more
          4. LSTM ensemble bias — shifts probability mass toward small (0-4) or big (5-9)
        """
        probs = np.ones(10) / 10.0  # uniform prior

        if len(digits) < 5:
            top = int(np.argmax(probs))
            return {
                "predicted": top,
                "confidence": 0.0,
                "probabilities": [round(float(p), 4) for p in probs],
            }

        # --- 1. Frequency-based probability (inverse = mean reversion) ---
        window = digits[-100:] if len(digits) > 100 else digits
        counts = np.zeros(10)
        for d in window:
            counts[d] += 1
        freq_probs = counts / counts.sum()
        
        # Blend: slight contrarian for overrepresented digits
        adjusted_freq = np.ones(10) / 10.0
        for i in range(10):
            deviation = freq_probs[i] - 0.1  # expected = 10%
            if abs(deviation) > 0.03:
                adjusted_freq[i] = 0.1 - deviation * 0.2
            else:
                adjusted_freq[i] = freq_probs[i]
        adjusted_freq = np.clip(adjusted_freq, 0.01, 1.0)
        adjusted_freq /= adjusted_freq.sum()

        # --- 2. Digit-level Markov transitions ---
        markov_probs = np.ones(10) / 10.0
        order = min(3, len(digits) - 1)
        if order >= 1:
            transitions = {}
            for i in range(order, len(digits)):
                state = tuple(digits[i - order:i])
                outcome = digits[i]
                if state not in transitions:
                    transitions[state] = np.zeros(10)
                transitions[state][outcome] += 1
            
            current_state = tuple(digits[-order:])
            if current_state in transitions:
                counts = transitions[current_state]
                total = counts.sum()
                if total > 0:
                    markov_probs = (counts + 0.5) / (total + 5.0)
                    markov_probs /= markov_probs.sum()

        # --- 3. Recency weighting (last 10 digits) ---
        recency_probs = np.ones(10) / 10.0
        recent = digits[-10:] if len(digits) >= 10 else digits
        rec_counts = np.zeros(10)
        for i, d in enumerate(recent):
            weight = (i + 1) / len(recent)
            rec_counts[d] += weight
        if rec_counts.sum() > 0:
            recency_probs = (rec_counts + 0.1) / (rec_counts.sum() + 1.0)
            recency_probs /= recency_probs.sum()

        # --- Combine all signals ---
        # Markov weight reduced (was 0.40) — it over-locks onto one digit per Markov
        # state, causing always-7 / always-1.  A 0.20 uniform floor ensures every
        # correct-side digit has baseline probability.
        base_combined = (
            0.30 * adjusted_freq +
            0.20 * markov_probs +
            0.30 * recency_probs +
            0.20 * (np.ones(10) / 10.0)
        )
        
        # --- The Hard Mask ---
        # Strictly enforce the ensemble's Big vs Small decision
        is_ensemble_big = ensemble_prob_big >= 0.5
        
        masked_combined = base_combined.copy()
        
        for i in range(10):
            if is_ensemble_big and i < 5:
                masked_combined[i] *= 0.25  # suppress wrong side (was 0.02 — too aggressive)
            elif not is_ensemble_big and i >= 5:
                masked_combined[i] *= 0.25  # suppress wrong side
                
        # Normalize the final masked probabilities
        combined = masked_combined / masked_combined.sum()

        # --- Digit diversity penalty ---
        # If the same digit has been predicted 2+ times in the last 4 rounds,
        # halve its probability so predictions rotate across the correct side.
        if len(self._last_digit_predictions) >= 2:
            recent_preds = list(self._last_digit_predictions)[-4:]
            for d in set(recent_preds):
                repeat_count = recent_preds.count(d)
                if repeat_count >= 2:
                    combined[d] *= max(0.25, 1.0 - 0.25 * repeat_count)
            combined = combined / combined.sum()

        top_digit = int(np.argmax(combined))

        # Post-hoc constraint: guarantee top_digit is always on the correct side.
        # The diversity penalty can temporarily depress the correct side enough for
        # a wrong-side digit to win argmax — clamp it back to the best correct-side digit.
        correct_range = list(range(5, 10)) if is_ensemble_big else list(range(0, 5))
        if top_digit not in correct_range:
            top_digit = max(correct_range, key=lambda d: combined[d])

        self._last_digit_predictions.append(top_digit)
        top_prob = float(combined[top_digit])
        
        # Confidence: how much better is the top digit vs uniform (10%)
        confidence = min((top_prob - 0.1) / 0.2, 1.0)
        confidence = max(confidence, 0.0)

        return {
            "predicted": top_digit,
            "confidence": round(float(confidence), 4),
            "probabilities": [round(float(p), 4) for p in combined],
        }

    def _predict_color(self, number_probs: list[float], predicted_digit: int = -1) -> dict:
        """
        Derive color prediction from the predicted digit, with probability
        distributions computed from number probabilities for informational bars.
        
        WinGo color rules:
          - Even numbers (0, 2, 4, 6, 8): Red
          - Odd numbers (1, 3, 7, 9): Green
          - 0: Red + Violet (special)
          - 5: Green + Violet (special)
        """
        probs = np.array(number_probs)
        
        # Probability distributions for the UI bars
        # Green: odd numbers (1, 3, 5, 7, 9)
        green_prob = float(probs[1] + probs[3] + probs[5] + probs[7] + probs[9])
        
        # Red: even numbers (0, 2, 4, 6, 8)
        red_prob = float(probs[0] + probs[2] + probs[4] + probs[6] + probs[8])
        
        # Violet: specifically digits 0 and 5
        violet_prob = float(probs[0] + probs[5])

        # Determine primary color DETERMINISTICALLY from the predicted digit
        if predicted_digit >= 0:
            if predicted_digit == 0:
                # 0 is Red + Violet — predict violet if violet probability is significant
                predicted = "violet" if violet_prob > 0.25 else "red"
            elif predicted_digit == 5:
                # 5 is Green + Violet — predict violet if violet probability is significant
                predicted = "violet" if violet_prob > 0.25 else "green"
            elif predicted_digit % 2 == 0:
                # Even (2, 4, 6, 8) = Red
                predicted = "red"
            else:
                # Odd (1, 3, 7, 9) = Green
                predicted = "green"
        else:
            # Fallback: no predicted digit, use probability distribution
            if violet_prob > 0.25 and violet_prob > max(green_prob, red_prob) * 0.5:
                predicted = "violet"
            elif green_prob > red_prob:
                predicted = "green"
            else:
                predicted = "red"

        # Confidence based on the predicted color's probability mass
        if predicted == "violet":
            conf = (violet_prob - 0.2) / 0.3
        elif predicted == "green":
            conf = green_prob - red_prob
        else:
            conf = red_prob - green_prob

        return {
            "predicted": predicted,
            "confidence": round(min(max(conf, 0.0), 1.0), 4),
            "probabilities": {
                "red": round(red_prob, 4),
                "green": round(green_prob, 4),
                "violet": round(violet_prob, 4),
            },
        }

    def record_outcome(self, actual_label: str):
        """Record prediction outcome for streak tracking and weight adjustment.
        Call this after the actual result is known."""
        if not self._pending_prediction:
            return

        actual_is_big = actual_label == "big"
        ensemble_correct = self._pending_prediction["ensemble_big"] == actual_is_big

        # Track per-model accuracy
        model_results = {}
        for name, pred_big in self._pending_prediction["models"].items():
            is_correct = pred_big == actual_is_big
            model_results[name] = is_correct
            self.model_total[name] = self.model_total.get(name, 0) + 1
            if is_correct:
                self.model_correct[name] = self.model_correct.get(name, 0) + 1

        # Track ensemble outcome for streak detection
        self.recent_outcomes.append({
            "ensemble": ensemble_correct,
            "models": model_results
        })

        self._pending_prediction = None

        # Live weight rebalancing every 5 outcomes once we have 10+
        if len(self.recent_outcomes) >= 10 and len(self.recent_outcomes) % 5 == 0:
            self._update_weights_live()

    def _get_loss_streak(self) -> int:
        """Get current consecutive loss count from recent outcomes."""
        streak = 0
        for outcome in reversed(self.recent_outcomes):
            if not outcome["ensemble"]:
                streak += 1
            else:
                break
        return streak

    def _update_weights_live(self):
        """Rebalance ensemble weights based on recent per-model accuracy."""
        if len(self.recent_outcomes) < 10:
            return

        # Use last 20 outcomes (or all available)
        window = list(self.recent_outcomes)[-20:]

        new_weights = {}
        for model_name in self.weights:
            correct_count = sum(1 for o in window if o["models"].get(model_name, False))
            total = len(window)
            acc = correct_count / total if total > 0 else 0.5
            # Weight = accuracy above chance (50%), with minimum floor
            new_weights[model_name] = max(0.05, acc - 0.3)

        # Normalize weights to sum to 1
        total_w = sum(new_weights.values())
        if total_w > 0:
            self.weights = {k: round(v / total_w, 4) for k, v in new_weights.items()}

    def _save_model(self):
        """Save model checkpoint."""
        if not self.lstm_model:
            return

        path = os.path.join(MODEL_DIR, f"best_lstm_{self.timer_name}.pt")
        torch.save({
            "model_state": self.lstm_model.state_dict(),
            "ensemble_weights": self.weights,
            "model_correct": self.model_correct,
            "model_total": self.model_total,
            "recent_outcomes": list(self.recent_outcomes),
            "last_digit_predictions": list(self._last_digit_predictions),
            "timestamp": time.time()
        }, path)
        print(f"[MODEL] Saved to {path}")

    def save(self):
        """Public save method."""
        self._save_model()

    def get_status(self) -> dict:
        """Get model status information."""
        return {
            "initialized": self._initialized,
            "timer": self.timer_name,
            "weights": self.weights,
            "markov_states": len(self.markov.transitions),
            "device": str(self.device)
        }
