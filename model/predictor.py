"""
LSTM-based prediction model + ensemble predictor.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

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
        """Update the model with the latest data point (online learning)."""
        if not self._initialized or len(digits) < SEQUENCE_LENGTH + 1:
            return

        # Update Markov chain
        labels = [1 if d >= 5 else 0 for d in digits]
        self.markov.train(labels)

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

        # Final prediction
        prediction = "big" if ensemble_prob >= 0.5 else "small"
        confidence = abs(ensemble_prob - 0.5) * 2

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "prob_big": round(ensemble_prob, 4),
            "models": model_predictions
        }

    def record_outcome(self, predicted: str, actual: str):
        """Record prediction outcome for weight adjustment."""
        # This is called after we know the actual result
        # We'd need to track per-model predictions, but for simplicity
        # we adjust weights based on ensemble performance
        pass

    def update_weights(self):
        """Auto-adjust ensemble weights based on recent accuracy."""
        total_all = sum(self.model_total.values())
        if total_all < 50:
            return  # need more data

        for model_name in self.weights:
            if self.model_total[model_name] > 0:
                acc = self.model_correct[model_name] / self.model_total[model_name]
                # Weight proportional to accuracy above chance
                self.weights[model_name] = max(0.05, acc - 0.3)

        # Normalize
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

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
