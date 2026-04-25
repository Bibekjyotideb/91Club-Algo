"""
Feature engineering pipeline for the prediction model.
Transforms raw result sequences into feature vectors.

Gap-aware: detects time gaps between results and marks them
so the LSTM doesn't learn fake patterns across gaps.

Time-aware: includes hour-of-day features since the game's
RNG may have time-based patterns.
"""
import numpy as np
import math
from collections import Counter, deque


class FeatureExtractor:
    """Extracts features from a sequence of results for model input."""

    def __init__(self, rolling_windows=None, markov_order=4):
        self.rolling_windows = rolling_windows or [5, 10, 20, 50]
        self.markov_order = markov_order

        # Expected intervals between results (seconds)
        # 30sec game = ~30s, 1min = ~60s, 3min = ~180s
        # We use a generous gap threshold: 5x expected interval
        self.gap_threshold = 300  # 5 minutes = definitely a gap for any timer

    def extract_features(self, digits: list[int], index: int,
                         timestamps: list[float] = None) -> np.ndarray:
        """
        Extract feature vector for a single position in the sequence.

        Features (16 total):
        1. Normalized digit value (0-1)
        2. Binary label (0=small, 1=big)
        3. Current streak length (normalized)
        4. Streak direction (0=small streak, 1=big streak)
        5-8. Rolling Big ratio for each window size
        9. Transition: did label change from previous?
        10. Frequency of current digit in last 20
        11-12. Last 2 labels as binary
        13. Gap marker: 1.0 if there was a significant time gap before this result
        14. Time since last result (normalized, capped)
        15-16. Hour of day as cyclical (sin, cos) — captures daily patterns
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

        # 13-14. Gap features (time-aware)
        if timestamps and index > 0 and index < len(timestamps):
            time_diff = timestamps[index] - timestamps[index - 1]
            is_gap = 1.0 if time_diff > self.gap_threshold else 0.0
            # Normalize time diff: 0 = instant, 1 = 10 minutes+
            norm_diff = min(time_diff / 600.0, 1.0)
        else:
            is_gap = 0.0
            norm_diff = 0.0

        features.append(is_gap)
        features.append(norm_diff)

        # 15-16. Hour of day as cyclical features
        if timestamps and index < len(timestamps):
            import time as _time
            local_time = _time.localtime(timestamps[index])
            hour = local_time.tm_hour + local_time.tm_min / 60.0
            features.append(math.sin(2 * math.pi * hour / 24.0))
            features.append(math.cos(2 * math.pi * hour / 24.0))
        else:
            features.append(0.0)
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def extract_sequence_features(self, digits: list[int], seq_length: int,
                                   timestamps: list[float] = None) -> tuple:
        """
        Extract feature sequences for LSTM training.
        Gap-aware: skips sequences that cross a time gap.

        Returns:
            X: array of shape (num_samples, seq_length, num_features)
            y: array of shape (num_samples,) — next label (0=small, 1=big)
        """
        if len(digits) <= seq_length:
            return np.array([]), np.array([])

        X_list = []
        y_list = []

        for i in range(seq_length, len(digits)):
            # Check for gap within this sequence window
            has_gap_in_window = False
            if timestamps:
                for k in range(i - seq_length + 1, i + 1):
                    if k > 0 and k < len(timestamps):
                        diff = timestamps[k] - timestamps[k - 1]
                        if diff > self.gap_threshold:
                            has_gap_in_window = True
                            break

            # Skip sequences that cross a gap — they don't represent
            # continuous game play and would teach wrong patterns
            if has_gap_in_window:
                continue

            seq_features = []
            for j in range(i - seq_length, i):
                feat = self.extract_features(digits, j, timestamps)
                seq_features.append(feat)

            X_list.append(np.array(seq_features))
            # Target: next result's label
            y_list.append(1.0 if digits[i] >= 5 else 0.0)

        if not X_list:
            return np.array([]), np.array([])

        return np.array(X_list), np.array(y_list)

    def extract_latest_sequence(self, digits: list[int], seq_length: int,
                                timestamps: list[float] = None) -> np.ndarray:
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
            if timestamps:
                # Pad timestamps with evenly spaced values
                if timestamps:
                    base = timestamps[0] if timestamps else 0
                    pad_ts = [base - (pad_needed - i) * 30 for i in range(pad_needed)]
                    timestamps = pad_ts + timestamps

        seq_features = []
        start = len(digits) - seq_length
        for j in range(start, len(digits)):
            feat = self.extract_features(digits, j, timestamps)
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
        return 16  # 12 original + gap_marker + time_diff + hour_sin + hour_cos


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
