"""
Feature engineering pipeline for the prediction model.
Transforms raw result sequences into feature vectors.
"""
import numpy as np
from collections import Counter, deque


class FeatureExtractor:
    """Extracts features from a sequence of results for model input."""

    def __init__(self, rolling_windows=None, markov_order=4):
        self.rolling_windows = rolling_windows or [5, 10, 20, 50]
        self.markov_order = markov_order

    def extract_features(self, digits: list[int], index: int) -> np.ndarray:
        """
        Extract feature vector for a single position in the sequence.

        Features:
        1. Normalized digit value (0-1)
        2. Binary label (0=small, 1=big)
        3. Current streak length (normalized)
        4. Streak direction (0=small streak, 1=big streak)
        5-8. Rolling Big ratio for each window size
        9. Transition: did label change from previous?
        10. Frequency of current digit in last 20
        11-12. Last 2 labels as binary
        """
        features = []

        digit = digits[index]
        label = 1 if digit >= 5 else 0

        # 1. Normalized digit
        features.append(digit / 9.0)

        # 2. Binary label
        features.append(float(label))

        # 3-4. Streak info
        streak_len, streak_dir = self._get_streak(digits, index)
        features.append(min(streak_len / 10.0, 1.0))  # cap at 10
        features.append(float(streak_dir))

        # 5-8. Rolling Big ratios
        for w in self.rolling_windows:
            start = max(0, index - w + 1)
            window = digits[start:index + 1]
            big_count = sum(1 for d in window if d >= 5)
            ratio = big_count / len(window) if window else 0.5
            features.append(ratio)

        # 9. Transition (label changed from previous?)
        if index > 0:
            prev_label = 1 if digits[index - 1] >= 5 else 0
            features.append(float(label != prev_label))
        else:
            features.append(0.5)

        # 10. Frequency of current digit in last 20
        start = max(0, index - 19)
        window = digits[start:index + 1]
        freq = window.count(digit) / len(window)
        features.append(freq)

        # 11-12. Last 2 labels
        for offset in [1, 2]:
            if index >= offset:
                prev = 1.0 if digits[index - offset] >= 5 else 0.0
                features.append(prev)
            else:
                features.append(0.5)

        return np.array(features, dtype=np.float32)

    def extract_sequence_features(self, digits: list[int], seq_length: int) -> tuple:
        """
        Extract feature sequences for LSTM training.

        Returns:
            X: array of shape (num_samples, seq_length, num_features)
            y: array of shape (num_samples,) — next label (0=small, 1=big)
        """
        if len(digits) <= seq_length:
            return np.array([]), np.array([])

        X_list = []
        y_list = []

        for i in range(seq_length, len(digits)):
            seq_features = []
            for j in range(i - seq_length, i):
                feat = self.extract_features(digits, j)
                seq_features.append(feat)

            X_list.append(np.array(seq_features))
            # Target: next result's label
            y_list.append(1.0 if digits[i] >= 5 else 0.0)

        return np.array(X_list), np.array(y_list)

    def extract_latest_sequence(self, digits: list[int], seq_length: int) -> np.ndarray:
        """
        Extract the latest sequence for prediction (no target needed).

        Returns:
            array of shape (1, seq_length, num_features)
        """
        if len(digits) < seq_length:
            # Pad with neutral values if not enough data
            pad_needed = seq_length - len(digits)
            padded = [5] * pad_needed + digits  # pad with neutral value
            digits = padded

        seq_features = []
        start = len(digits) - seq_length
        for j in range(start, len(digits)):
            feat = self.extract_features(digits, j)
            seq_features.append(feat)

        return np.array([seq_features], dtype=np.float32)

    def _get_streak(self, digits: list[int], index: int) -> tuple:
        """Get current streak length and direction at given index."""
        if index < 0:
            return 0, 0

        current_label = 1 if digits[index] >= 5 else 0
        streak = 1

        for i in range(index - 1, -1, -1):
            prev_label = 1 if digits[i] >= 5 else 0
            if prev_label == current_label:
                streak += 1
            else:
                break

        return streak, current_label

    @property
    def num_features(self) -> int:
        """Number of features per timestep."""
        return 12  # Update if you add more features


class MarkovChain:
    """Markov chain model for transition probability estimation."""

    def __init__(self, order: int = 4):
        self.order = order
        self.transitions = {}  # state -> {0: count, 1: count}

    def train(self, labels: list[int]):
        """Train on a sequence of binary labels (0=small, 1=big)."""
        self.transitions = {}
        for i in range(self.order, len(labels)):
            state = tuple(labels[i - self.order:i])
            outcome = labels[i]

            if state not in self.transitions:
                self.transitions[state] = {0: 0, 1: 0}
            self.transitions[state][outcome] += 1

    def predict(self, recent_labels: list[int]) -> tuple:
        """
        Predict next outcome probability.

        Returns:
            (probability_of_big, confidence) or (0.5, 0.0) if state unseen
        """
        if len(recent_labels) < self.order:
            return 0.5, 0.0

        state = tuple(recent_labels[-self.order:])

        if state not in self.transitions:
            return 0.5, 0.0

        counts = self.transitions[state]
        total = counts[0] + counts[1]

        if total == 0:
            return 0.5, 0.0

        prob_big = counts[1] / total
        # Confidence based on sample size (more samples = more confident)
        confidence = min(total / 20.0, 1.0)  # cap at 20 observations

        return prob_big, confidence


class PatternDetector:
    """Detects repeating patterns in the result sequence."""

    def __init__(self, max_pattern_length: int = 8):
        self.max_pattern_length = max_pattern_length

    def detect_patterns(self, labels: list[int]) -> dict:
        """
        Detect repeating patterns and estimate next outcome.

        Returns:
            {"prob_big": float, "confidence": float, "patterns_found": int}
        """
        if len(labels) < 4:
            return {"prob_big": 0.5, "confidence": 0.0, "patterns_found": 0}

        votes_big = 0
        votes_small = 0
        patterns_found = 0

        for plen in range(2, min(self.max_pattern_length + 1, len(labels) // 2)):
            current = tuple(labels[-plen:])

            # Search for this pattern in history
            for i in range(len(labels) - plen - 1):
                candidate = tuple(labels[i:i + plen])
                if candidate == current and i + plen < len(labels):
                    # What followed this pattern?
                    next_val = labels[i + plen]
                    if next_val == 1:
                        votes_big += 1
                    else:
                        votes_small += 1
                    patterns_found += 1

        total_votes = votes_big + votes_small
        if total_votes == 0:
            return {"prob_big": 0.5, "confidence": 0.0, "patterns_found": 0}

        prob_big = votes_big / total_votes
        confidence = min(total_votes / 15.0, 1.0)

        return {"prob_big": prob_big, "confidence": confidence, "patterns_found": patterns_found}


class FrequencyAnalyzer:
    """Analyzes frequency distribution for bias detection."""

    def analyze(self, digits: list[int], window: int = 100) -> dict:
        """
        Check if outcomes deviate from expected 50/50.

        Returns:
            {"prob_big": float, "confidence": float, "bias": float}
        """
        if len(digits) < 10:
            return {"prob_big": 0.5, "confidence": 0.0, "bias": 0.0}

        recent = digits[-window:] if len(digits) > window else digits
        big_count = sum(1 for d in recent if d >= 5)
        total = len(recent)
        actual_ratio = big_count / total

        # If there's a bias, predict the more frequent outcome
        # But the prediction should account for potential mean reversion
        bias = actual_ratio - 0.5

        # If significantly biased, might mean-revert or continue
        # Use a mild contrarian approach for strong biases
        if abs(bias) > 0.15:
            # Strong bias — slight contrarian
            prob_big = 0.5 - (bias * 0.3)
        else:
            # Mild bias — follow the trend
            prob_big = actual_ratio

        confidence = min(total / 50.0, 1.0) * min(abs(bias) * 5, 1.0)

        return {"prob_big": prob_big, "confidence": confidence, "bias": bias}
