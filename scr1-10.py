import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from collections import Counter, defaultdict
import warnings
import os
import json
import traceback
from pathlib import Path
import math
import calendar
import random
import hashlib
from quant_excel_loader import load_results_excel
from quant_data_core import compute_learning_signals, apply_learning_to_dataframe

import sys

QUIET_MODE = '--quiet' in sys.argv

def quiet_print(*args, **kwargs):
    if not QUIET_MODE:
        print(*args, **kwargs)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

class UltimateAllPredictor:
    def __init__(self):
        self.slot_names = {1: "FRBD", 2: "GZBD", 3: "GALI", 4: "DSWR"}
        self.slot_name_to_id = {name: sid for sid, name in self.slot_names.items()}
        
        self.PROJECT_DIR = Path(__file__).resolve().parent
        self.COMBINED_DIR = self.PROJECT_DIR / "predictions" / "ultimate_all"
        self.COMBINED_DIR.mkdir(parents=True, exist_ok=True)
        
        self.WEIGHTS_FILE = self.PROJECT_DIR / "model_weights.json"
        self.PNL_FILE = self.PROJECT_DIR / "pnl.xlsx"
        self.PERFORMANCE_FILE = self.PROJECT_DIR / "performance_tracker.json"
        
        self.stake_per_number = 10
        self.andar_stake_per_digit = 10
        self.bahar_stake_per_digit = 10
        
        self.initialize_pattern_packs()
        self.model_weights = {}
        self.processed_dates = set()
        self.load_weights_from_json()
        self.load_performance_tracker()
        
        self.lstm_models = {}
        self.rf_models = {}
        self.gb_models = {}
        self.mlp_models = {}
        self.xgb_models = {}
        self.performance_history = defaultdict(list)
        
        self.slot_classification = {"FRBD": "neutral", "GZBD": "neutral", "GALI": "neutral", "DSWR": "neutral"}
        self.slot_roi = {"FRBD": 0.0, "GZBD": 0.0, "GALI": 0.0, "DSWR": 0.0}
        self.slot_win_streak = {"FRBD": 0, "GZBD": 0, "DSWR": 0, "GALI": 0}
        self.slot_loss_streak = {"FRBD": 0, "GZBD": 0, "DSWR": 0, "GALI": 0}
        
        # OPTIMIZED STRATEGIES
        self.slot_strategies = {
            'FRBD': {'focus_range': range(0, 34), 'optimal_k': 18, 'boost_hot': True, 
                    'stake_multiplier': 1.2, 'max_k': 25, 'min_k': 12, 'unit_stake': 10},
            'GZBD': {'focus_range': range(34, 67), 'optimal_k': 15, 'boost_patterns': True,
                    'stake_multiplier': 0.5, 'max_k': 20, 'min_k': 5, 'protection_mode': True, 'unit_stake': 5},
            'GALI': {'focus_range': range(0, 100), 'optimal_k': 20, 'boost_hot': False,
                    'stake_multiplier': 0.9, 'max_k': 30, 'min_k': 15, 'unit_stake': 10},
            'DSWR': {'focus_range': range(67, 100), 'optimal_k': 18, 'boost_s40': True,
                    'stake_multiplier': 1.3, 'max_k': 25, 'min_k': 12, 'unit_stake': 10}
        }
        
        self.optimal_top_k = {'FRBD': 18, 'GZBD': 15, 'GALI': 20, 'DSWR': 18}
        self.dynamic_stakes = {'FRBD': 10, 'GZBD': 10, 'GALI': 10, 'DSWR': 10}
        self.conservative_mode = False
        self.emergency_mode = False
        self.total_profit = 0
        self.total_days = 0
        self.min_weight = 0.10
        self.max_weight = 0.80
        
        # DATA LEAKAGE PREVENTION
        self.last_validation_date = None
        self.training_window = 90
        self.validation_gap = 2
        
    def initialize_pattern_packs(self):
        self.S40_numbers = {
            0,6,7,9,15,16,18,19,24,25,27,28,33,34,36,37,
            42,43,45,46,51,52,54,55,60,61,63,64,70,72,73,
            79,81,82,88,89,90,91,97,98
        }
        self.SLOT_HOT = {"FRBD": [], "GZBD": [], "GALI": [], "DSWR": []}
        self.SLOT_COLD = {"FRBD": [], "GZBD": [], "GALI": [], "DSWR": []}
        self.GLOBAL_MULTI_SLOT_HOT = []
        
        self.pattern_3digit = self._generate_pattern_sets(3)
        self.pattern_4digit = self._generate_pattern_sets(4)
        self.pattern_6digit = self._generate_pattern_sets(6)
        self.last_10_results = {slot: [] for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']}
    
    def _generate_pattern_sets(self, length):
        patterns = {}
        for start in range(10):
            digits = [(start + i) % 10 for i in range(length)]
            pattern_name = ''.join(map(str, digits))
            pattern_set = set()
            for tens in digits:
                for ones in digits:
                    num = tens * 10 + ones
                    if 0 <= num <= 99: pattern_set.add(num)
            patterns[pattern_name] = pattern_set
        return patterns
    
    def load_performance_tracker(self):
        if self.PERFORMANCE_FILE.exists():
            try:
                with open(self.PERFORMANCE_FILE, 'r') as f:
                    data = json.load(f)
                self.total_profit = data.get('total_profit', 0)
                self.total_days = data.get('total_days', 0)
                self.slot_win_streak = data.get('slot_win_streak', self.slot_win_streak)
                self.slot_loss_streak = data.get('slot_loss_streak', self.slot_loss_streak)
            except: pass
    
    def save_performance_tracker(self):
        try:
            data = {
                'total_profit': self.total_profit,
                'total_days': self.total_days,
                'slot_win_streak': self.slot_win_streak,
                'slot_loss_streak': self.slot_loss_streak,
                'last_update': datetime.now().strftime('%Y-%m-%d')
            }
            with open(self.PERFORMANCE_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except: pass
    
    def load_weights_from_json(self):
        if self.WEIGHTS_FILE.exists():
            try:
                with open(self.WEIGHTS_FILE, 'r') as f:
                    data = json.load(f)
                self.model_weights = data.get('weights', {})
                self.processed_dates = set()
            except:
                self.initialize_default_weights()
        else:
            self.initialize_default_weights()
    
    def save_weights_to_json(self):
        try:
            data = {'weights': self.model_weights, 'processed_dates': []}
            with open(self.WEIGHTS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except: pass
    
    def initialize_default_weights(self):
        scripts = ['scr1', 'scr2', 'scr3', 'scr4', 'scr6', 'scr7', 'scr8', 'scr9', 'scr10', 'scr11', 'scr12', 'scr13', 'scr14']
        for script in scripts:
            for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                model_id = f"{script}_{slot}"
                self.model_weights[model_id] = {
                    'weight': 0.3, 'overall_accuracy': 0.0, 'total_hits': 0, 'total_attempts': 0,
                    'weight_history': [0.3], 'performance_history': [],
                    'recent_hits': 0, 'recent_attempts': 0, 'consistency_score': 0.5
                }
        self.save_weights_to_json()
    
    # ================== EXISTING SCRIPTS 1-12 (PRESERVED) ==================
    
    def generate_scr1_predictions(self, df, target_date, top_k=10):
        lookback = min(60, len(df))
        comps = self._scr1_build_components(df.tail(lookback), halflife_slot=30.0)
        dow = pd.Timestamp(target_date).dayofweek
        rows = []
        for s in [1, 2, 3, 4]:
            sc = self._scr1_scores_for_slot(comps, dow, s)
            picks = sorted(sc.items(), key=lambda x: x[1], reverse=True)[:top_k]
            for rank, (num, score) in enumerate(picks, start=1):
                rows.append({
                    'date': target_date.strftime('%Y-%m-%d'), 'slot': s, 'rank': rank,
                    'number': f"{int(num)%100:02d}", 'source': 'scr1'
                })
        return pd.DataFrame(rows)
    
    def _scr1_ewma_weights(self, n, halflife=30.0):
        if n <= 0: return []
        idx = np.arange(n)
        w = 0.5 ** ((n-1 - idx)/halflife)
        s = w.sum()
        return (w / s) if s > 0 else np.ones(n)/n
    
    def _scr1_normalize(self, scores):
        if not scores: return {}
        arr = np.array(list(scores.values()), dtype=float)
        if np.all(arr == 0): return {k: 0.0 for k in scores}
        mn, mx = float(arr.min()), float(arr.max())
        if mx - mn < 1e-12: return {k: 0.0 for k in scores}
        return {k: (float(v)-mn)/(mx-mn) for k, v in scores.items()}
    
    def _scr1_build_components(self, df, halflife_slot=30.0):
        comps = {'rec_slot': {}, 'last_seen': {}, 'trans_prob': {}, 'dow_bonus': {}}
        for s in [1, 2, 3, 4]:
            sub = df[df['slot']==s].sort_values('date').reset_index(drop=True)
            if len(sub) == 0:
                comps['rec_slot'][s] = {n: 0.0 for n in range(100)}
                continue
            ws = self._scr1_ewma_weights(len(sub), halflife=halflife_slot)
            sub['w'] = ws
            rec = {n: float(sub.loc[sub['number']==n, 'w'].sum()) for n in range(100)}
            comps['rec_slot'][s] = self._scr1_normalize(rec)
            comps['last_seen'][s] = int(sub.loc[len(sub)-1, 'number'])
            counts = {i: {j: 1e-3 for j in range(100)} for i in range(100)}
            for k in range(1, len(sub)):
                prev_n = int(sub.loc[k-1, 'number'])
                curr_n = int(sub.loc[k, 'number'])
                counts[prev_n][curr_n] += 1.0
            trans = {}
            for i in range(100):
                row = counts[i]
                ssum = sum(row.values())
                trans[i] = {j: (row[j]/ssum) for j in range(100)} if ssum>0 else {j: 0.0 for j in range(100)}
            comps['trans_prob'][s] = trans
        df = df.copy()
        df['dow'] = df['date'].dt.dayofweek
        for d in range(7):
            sub = df[df['dow']==d]
            if len(sub) == 0:
                comps['dow_bonus'][d] = {n: 0.0 for n in range(100)}
                continue
            vc = sub['number'].value_counts()
            mx = vc.max()
            comps['dow_bonus'][d] = {int(n): (c/mx) for n, c in vc.items()}
            for n in range(100): comps['dow_bonus'][d].setdefault(n, 0.0)
        return comps
    
    def _scr1_scores_for_slot(self, comps, target_dow, slot):
        rec = comps['rec_slot'].get(slot, {n: 0.0 for n in range(100)})
        lastn = comps['last_seen'].get(slot, None)
        if lastn is None:
            trans = {n: 0.0 for n in range(100)}
        else:
            trans_row = comps['trans_prob'][slot][lastn]
            trans = self._scr1_normalize(trans_row)
        dow_s = comps['dow_bonus'].get(target_dow, {n: 0.0 for n in range(100)})
        scores = {}
        for n in range(100):
            scores[n] = 0.5*rec.get(n, 0.0) + 0.35*trans.get(n, 0.0) + 0.15*dow_s.get(n, 0.0)
        return scores
    
    def generate_scr2_predictions(self, df, target_date, top_k=10):
        rows = []
        for slot in [1, 2, 3, 4]:
            slot_data = df[df['slot'] == slot]
            lookback = min(50, len(slot_data))
            slot_pred = self._scr2_ensemble_scoring(slot_data.tail(lookback), slot, top_k)
            for rank, (number, confidence) in enumerate(slot_pred, 1):
                rows.append({
                    'date': target_date.strftime('%Y-%m-%d'), 'slot': slot, 'rank': rank,
                    'number': f"{number:02d}", 'source': 'scr2'
                })
        return pd.DataFrame(rows)
    
    def _scr2_ensemble_scoring(self, df, slot, top_k=15):
        slot_data = df[df['slot'] == slot]
        numbers = slot_data['number'].tolist()
        if len(numbers) < 10:
            freq = Counter(numbers)
            return [(num, freq.get(num, 0)/len(numbers)) for num in range(100)][:top_k]
        weights = np.exp(np.linspace(0, 1, min(60, len(numbers))))
        weights = weights / weights.sum()
        weighted_freq = {}
        for i, num in enumerate(numbers[-60:]):
            weight = weights[i] if i < len(weights) else 1.0
            weighted_freq[num] = weighted_freq.get(num, 0) + weight
        scores = {}
        for num in range(100):
            score = weighted_freq.get(num, 0) * 0.4
            positions = [i for i, n in enumerate(numbers) if n == num]
            if len(positions) > 1:
                gaps = [positions[i] - positions[i-1] for i in range(1, len(positions))]
                avg_gap = np.mean(gaps)
                current_gap = len(numbers) - positions[-1]
                score += min(current_gap / avg_gap, 3.0) * 0.25 if avg_gap > 0 else 0
            if numbers:
                transitions = {}
                for i in range(1, len(numbers)):
                    prev, curr = numbers[i-1], numbers[i]
                    if prev not in transitions: transitions[prev] = {}
                    transitions[prev][curr] = transitions[prev].get(curr, 0) + 1
                if numbers[-1] in transitions and num in transitions[numbers[-1]]:
                    total = sum(transitions[numbers[-1]].values())
                    score += (transitions[numbers[-1]][num] / total) * 0.15
            if score > 0: scores[num] = score
        if scores:
            max_score = max(scores.values())
            if max_score > 0: scores = {num: score/max_score for num, score in scores.items()}
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    def generate_scr3_predictions(self, df, target_date, top_k=10):
        rows = []
        for slot in [1, 2, 3, 4]:
            slot_data = df[df['slot'] == slot]
            lookback = min(40, len(slot_data))
            numbers = slot_data.tail(lookback)['number'].tolist()
            pred_numbers = self._scr3_ensemble_prediction(numbers, top_k)
            for rank, number in enumerate(pred_numbers, 1):
                rows.append({
                    'date': target_date.strftime('%Y-%m-%d'), 'slot': slot, 'rank': rank,
                    'number': f"{number:02d}", 'source': 'scr3'
                })
        return pd.DataFrame(rows)
    
    def _scr3_ensemble_prediction(self, numbers, top_k=15):
        if len(numbers) < 10:
            freq = Counter(numbers)
            return [num for num, count in freq.most_common(top_k)]
        window = min(30, len(numbers))
        recent_data = numbers[-window:]
        weights = np.exp(np.linspace(0, 1, window))
        weights = weights / weights.sum()
        number_counts = {}
        for idx, num in enumerate(recent_data):
            weight = weights[idx] if idx < len(weights) else 1.0
            number_counts[num] = number_counts.get(num, 0) + weight
        freq_pred = [num for num, _ in sorted(number_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]]
        positions = {}
        for i, num in enumerate(numbers):
            if num not in positions: positions[num] = []
            positions[num].append(i)
        gap_scores = {}
        current_idx = len(numbers) - 1
        for num in range(100):
            if num in positions and len(positions[num]) > 1:
                gaps = [positions[num][i] - positions[num][i-1] for i in range(1, len(positions[num]))]
                avg_gap = np.mean(gaps)
                current_gap = current_idx - positions[num][-1]
                gap_scores[num] = current_gap / avg_gap if avg_gap > 0 else 10.0
            else:
                gap_scores[num] = 10.0
        gap_pred = [num for num, _ in sorted(gap_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]
        combined_scores = defaultdict(float)
        for rank, num in enumerate(freq_pred):
            combined_scores[num] += 0.5 * (len(freq_pred) - rank) / len(freq_pred)
        for rank, num in enumerate(gap_pred):
            combined_scores[num] += 0.5 * (len(gap_pred) - rank) / len(gap_pred)
        final_predictions = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in final_predictions[:top_k]]
    
    def generate_scr4_predictions(self, df, target_date, top_k=10):
        rows = []
        for slot in [1, 2, 3, 4]:
            slot_data = df[df['slot'] == slot]
            lookback = min(50, len(slot_data))
            numbers = slot_data.tail(lookback)['number'].tolist()
            pred_numbers = self._scr4_advanced_ensemble(numbers, top_k)
            for rank, number in enumerate(pred_numbers, 1):
                rows.append({
                    'date': target_date.strftime('%Y-%m-%d'), 'slot': slot, 'rank': rank,
                    'number': f"{number:02d}", 'source': 'scr4'
                })
        return pd.DataFrame(rows)
    
    def _scr4_bayesian_probability(self, numbers, top_k):
        windows = [20, 30, 40, 50]
        alpha = 1.0
        combined_probs = {}
        for window in windows:
            if len(numbers) >= window:
                recent = numbers[-window:]
                freq = Counter(recent)
                total = len(recent)
                for num in range(100):
                    count = freq.get(num, 0)
                    prob = (count + alpha) / (total + 100 * alpha)
                    combined_probs[num] = combined_probs.get(num, 0) + prob
        if not combined_probs:
            freq = Counter(numbers)
            total = len(numbers)
            for num in range(100):
                count = freq.get(num, 0)
                combined_probs[num] = (count + alpha) / (total + 100 * alpha)
        sorted_probs = sorted(combined_probs.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_probs[:top_k]]
    
    def _scr4_confidence_gap_analysis(self, numbers, top_k):
        positions = {}
        for i, num in enumerate(numbers):
            if num not in positions: positions[num] = []
            positions[num].append(i)
        gap_scores = {}
        current_idx = len(numbers) - 1
        for num in range(100):
            if num in positions and len(positions[num]) > 2:
                gaps = [positions[num][i] - positions[num][i-1] for i in range(1, len(positions[num]))]
                avg_gap = np.mean(gaps)
                std_gap = np.std(gaps)
                current_gap = current_idx - positions[num][-1]
                if current_gap > avg_gap:
                    confidence = min((current_gap - avg_gap) / (std_gap + 1), 3.0) / 3.0
                else:
                    confidence = 0.1
                gap_scores[num] = confidence
            else:
                gap_scores[num] = 0.5
        due_numbers = sorted(gap_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in due_numbers[:top_k]]
    
    def _scr4_advanced_pattern_mining(self, numbers, top_k):
        sequences = defaultdict(list)
        for length in [2, 3, 4]:
            for i in range(len(numbers) - length):
                seq = tuple(numbers[i:i+length])
                next_val = numbers[i+length]
                sequences[seq].append(next_val)
        predictions = []
        for length in [4, 3, 2]:
            if len(numbers) >= length:
                recent_seq = tuple(numbers[-length:])
                if recent_seq in sequences:
                    next_vals = sequences[recent_seq]
                    counter = Counter(next_vals)
                    predictions.extend([num for num, _ in counter.most_common(3)])
        if not predictions:
            freq = Counter(numbers[-20:])
            predictions = [num for num, _ in freq.most_common(top_k)]
        return predictions[:top_k]
    
    def _scr4_advanced_ensemble(self, numbers, top_k=15):
        if len(numbers) < 10:
            freq = Counter(numbers)
            return [num for num, _ in freq.most_common(top_k)]
        strategies = {
            'bayesian': (self._scr4_bayesian_probability(numbers, top_k), 0.25),
            'gap_analysis': (self._scr4_confidence_gap_analysis(numbers, top_k), 0.20),
            'patterns': (self._scr4_advanced_pattern_mining(numbers, top_k), 0.20),
        }
        final_scores = defaultdict(float)
        for strategy_name, (predictions, weight) in strategies.items():
            for rank, num in enumerate(predictions):
                position_weight = (len(predictions) - rank) / len(predictions)
                final_scores[num] += weight * position_weight
        final_predictions = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in final_predictions[:top_k]]
    
    def generate_scr5_predictions(self, df, target_date, top_k=10):
        return self.generate_scr4_predictions(df, target_date, top_k)
    
    def build_lstm_model(self, lookback=30):
        if lookback in self.lstm_models:
            return self.lstm_models[lookback]
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=(lookback, 1)),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        self.lstm_models[lookback] = model
        return model
    
    def generate_scr6_predictions(self, df, target_date, top_k=10):
        rows = []
        for slot in [1, 2, 3, 4]:
            slot_data = df[df['slot'] == slot]
            lookback = min(60, len(slot_data))
            numbers = slot_data.tail(lookback)['number'].tolist()
            pred_numbers = self._scr6_advanced_ensemble(numbers, top_k)
            for rank, number in enumerate(pred_numbers, 1):
                rows.append({
                    'date': target_date.strftime('%Y-%m-%d'), 'slot': slot, 'rank': rank,
                    'number': f"{number:02d}", 'source': 'scr6'
                })
        return pd.DataFrame(rows)
    
    def _scr6_advanced_ensemble(self, numbers, top_k=15):
        if len(numbers) < 30:
            freq = Counter(numbers)
            return [num for num, _ in freq.most_common(top_k)]
        strategies = {}
        if len(numbers) >= 60:
            try:
                lookback = 30
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(np.array(numbers).reshape(-1, 1))
                X, y = [], []
                for i in range(lookback, len(scaled_data)):
                    X.append(scaled_data[i-lookback:i, 0])
                    y.append(scaled_data[i, 0])
                X, y = np.array(X), np.array(y)
                X = X.reshape(X.shape[0], X.shape[1], 1)
                split_idx = int(0.8 * len(X))
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                model = self.build_lstm_model(lookback)
                early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)
                last_sequence = scaled_data[-lookback:]
                last_sequence = last_sequence.reshape(1, lookback, 1)
                next_pred_scaled = model.predict(last_sequence, verbose=0)
                next_pred = scaler.inverse_transform(next_pred_scaled)[0, 0]
                next_pred = int(np.clip(np.round(next_pred), 0, 99))
                base_predictions = []
                for offset in range(-top_k//2, top_k//2 + 1):
                    pred_num = (next_pred + offset) % 100
                    if pred_num not in base_predictions:
                        base_predictions.append(pred_num)
                strategies['lstm'] = (base_predictions[:top_k], 0.15)
            except:
                pass
        strategies['bayesian'] = (self._scr4_bayesian_probability(numbers, top_k), 0.15)
        strategies['gap_analysis'] = (self._scr4_confidence_gap_analysis(numbers, top_k), 0.15)
        strategies['patterns'] = (self._scr4_advanced_pattern_mining(numbers, top_k), 0.10)
        final_scores = defaultdict(float)
        for strategy_name, (predictions, weight) in strategies.items():
            reliability = 0.9 * min(len(numbers) / 100, 1.0)
            adjusted_weight = weight * reliability
            for rank, num in enumerate(predictions):
                position_weight = (len(predictions) - rank) / len(predictions)
                final_scores[num] += adjusted_weight * position_weight
        final_predictions = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        predictions_list = [num for num, _ in final_predictions[:top_k*2]]
        ranges = {'low': [n for n in predictions_list if 0 <= n <= 33],
                 'medium': [n for n in predictions_list if 34 <= n <= 66],
                 'high': [n for n in predictions_list if 67 <= n <= 99]}
        selected = []
        for range_name in ['low', 'medium', 'high']:
            if ranges[range_name]: selected.append(ranges[range_name][0])
        s40_candidates = [n for n in predictions_list if n in self.S40_numbers and n not in selected]
        if s40_candidates and len(selected) < top_k:
            selected.append(s40_candidates[0])
        remaining = top_k - len(selected)
        if remaining > 0:
            for num in predictions_list:
                if num not in selected and len(selected) < top_k:
                    selected.append(num)
        return selected[:top_k]
    
    def generate_scr7_predictions(self, df, target_date, top_k=10):
        rows = []
        for slot in [1, 2, 3, 4]:
            slot_name = self.slot_names[slot]
            slot_data = df[df['slot'] == slot]
            lookback = min(50, len(slot_data))
            numbers = slot_data.tail(lookback)['number'].tolist()
            
            freq_pred = self._scr4_bayesian_probability(numbers, 10)
            gap_pred = self._scr4_confidence_gap_analysis(numbers, 10)
            pattern_pred = self._scr4_advanced_pattern_mining(numbers, 10)
            
            combined = Counter()
            for rank, num in enumerate(freq_pred):
                combined[num] += 0.4 * (10 - rank) / 10
            for rank, num in enumerate(gap_pred):
                combined[num] += 0.3 * (10 - rank) / 10
            for rank, num in enumerate(pattern_pred):
                combined[num] += 0.3 * (10 - rank) / 10
            
            pred_numbers = [num for num, _ in combined.most_common(top_k)]
            
            for rank, number in enumerate(pred_numbers, 1):
                rows.append({
                    'date': target_date.strftime('%Y-%m-%d'), 'slot': slot, 'rank': rank,
                    'number': f"{number:02d}", 'source': 'scr7'
                })
        return pd.DataFrame(rows)
    
    def generate_scr8_predictions(self, df, target_date, top_k=10):
        rows = []
        for slot in [1, 2, 3, 4]:
            slot_name = self.slot_names[slot]
            slot_data = df[df['slot'] == slot]
            lookback = min(60, len(slot_data))
            numbers = slot_data.tail(lookback)['number'].tolist()
            
            if len(numbers) < 10:
                freq = Counter(numbers)
                pred_numbers = [num for num, _ in freq.most_common(top_k)]
            else:
                base_scores = Counter()
                bayesian_pred = self._scr4_bayesian_probability(numbers, 20)
                for rank, num in enumerate(bayesian_pred):
                    base_scores[num] += 0.4 * (20 - rank) / 20
                gap_pred = self._scr4_confidence_gap_analysis(numbers, 20)
                for rank, num in enumerate(gap_pred):
                    base_scores[num] += 0.3 * (20 - rank) / 20
                pattern_pred = self._scr4_advanced_pattern_mining(numbers, 20)
                for rank, num in enumerate(pattern_pred):
                    base_scores[num] += 0.3 * (20 - rank) / 20
                pattern_scores = self._scr8_pattern_based_scoring(slot_name, numbers, base_scores)
                pred_numbers = [num for num, _ in pattern_scores.most_common(top_k)]
            
            for rank, number in enumerate(pred_numbers, 1):
                rows.append({
                    'date': target_date.strftime('%Y-%m-%d'), 'slot': slot, 'rank': rank,
                    'number': f"{number:02d}", 'source': 'scr8'
                })
        return pd.DataFrame(rows)
    
    def get_opposite(self, n):
        if n < 10: return n * 10
        else:
            tens = n // 10
            ones = n % 10
            return ones * 10 + tens
    
    def _scr8_pattern_based_scoring(self, slot_name, numbers, base_models_score):
        scores = Counter()
        for num in range(100):
            score = 0.0
            score += base_models_score.get(num, 0.0)
            if num in self.SLOT_HOT[slot_name]: score += 2.0
            if num in self.SLOT_COLD[slot_name]: score -= 1.0
            if num in self.GLOBAL_MULTI_SLOT_HOT: score += 1.0
            if num in self.S40_numbers: score += 0.5
            opposite_num = self.get_opposite(num)
            for other_slot, hot_list in self.SLOT_HOT.items():
                if other_slot == slot_name: continue
                if opposite_num in hot_list: score += 0.7
            if score > 0: scores[num] = score
        return scores
    
    def generate_scr9_predictions(self, df, target_date, top_k=10):
        rows = []
        for slot in [1, 2, 3, 4]:
            slot_name = self.slot_names[slot]
            slot_data = df[df['slot'] == slot]
            lookback = min(50, len(slot_data))
            numbers = slot_data.tail(lookback)['number'].tolist()
            
            if len(numbers) < 20:
                freq = Counter(numbers)
                pred_numbers = [num for num, _ in freq.most_common(top_k)]
            else:
                n_simulations = 1000
                predictions = []
                for _ in range(n_simulations):
                    transitions = {}
                    for i in range(1, len(numbers)):
                        prev, curr = numbers[i-1], numbers[i]
                        if prev not in transitions: transitions[prev] = []
                        transitions[prev].append(curr)
                    current = numbers[-1]
                    path = [current]
                    for step in range(5):
                        if current in transitions and transitions[current]:
                            next_num = np.random.choice(transitions[current])
                            path.append(next_num)
                            current = next_num
                        else: break
                    predictions.append(path[-1] if len(path) > 1 else numbers[-1])
                cross_correlation = Counter()
                for other_slot in [1, 2, 3, 4]:
                    if other_slot == slot: continue
                    other_data = df[df['slot'] == other_slot]
                    other_numbers = other_data['number'].tolist()
                    if len(other_numbers) > 10:
                        for i in range(min(len(numbers), len(other_numbers))):
                            if numbers[i] == other_numbers[i]:
                                cross_correlation[numbers[i]] += 1
                mc_counter = Counter(predictions)
                combined_scores = Counter()
                for num in range(100):
                    score = mc_counter.get(num, 0) / n_simulations
                    score += cross_correlation.get(num, 0) * 0.1
                    for other_slot_name, hot_list in self.SLOT_HOT.items():
                        if other_slot_name != slot_name and num in hot_list:
                            score += 0.3
                    if score > 0: combined_scores[num] = score
                pred_numbers = [num for num, _ in combined_scores.most_common(top_k)]
            
            for rank, number in enumerate(pred_numbers, 1):
                rows.append({
                    'date': target_date.strftime('%Y-%m-%d'), 'slot': slot, 'rank': rank,
                    'number': f"{number:02d}", 'source': 'scr9'
                })
        return pd.DataFrame(rows)
    
    def generate_scr10_predictions(self, df, target_date, top_k=10):
        rows = []
        for slot in [1, 2, 3, 4]:
            slot_name = self.slot_names[slot]
            slot_data = df[df['slot'] == slot]
            lookback = min(50, len(slot_data))
            numbers = slot_data.tail(lookback)['number'].tolist()
            
            if len(numbers) < 30:
                freq = Counter(numbers[-20:])
                pred_numbers = [num for num, _ in freq.most_common(top_k)]
            else:
                population_size = 50
                generations = 20
                mutation_rate = 0.1
                population = []
                for _ in range(population_size):
                    if random.random() < 0.3:
                        individual = random.sample(range(100), top_k)
                    elif random.random() < 0.6:
                        individual = []
                        hot_nums = self.SLOT_HOT[slot_name][:min(10, len(self.SLOT_HOT[slot_name]))]
                        individual.extend(hot_nums)
                        s40_list = list(self.S40_numbers)
                        if len(s40_list) > 0:
                            individual.extend(random.sample(s40_list, min(5, len(s40_list))))
                        while len(individual) < top_k:
                            num = random.randint(0, 99)
                            if num not in individual: individual.append(num)
                    else:
                        freq_counts = Counter(numbers[-50:])
                        individual = [num for num, _ in freq_counts.most_common(top_k)]
                    population.append(individual[:top_k])
                for generation in range(generations):
                    fitness_scores = []
                    for individual in population:
                        fitness = self._evaluate_ga_fitness(individual, numbers, top_k)
                        fitness_scores.append((individual, fitness))
                    fitness_scores.sort(key=lambda x: x[1], reverse=True)
                    elite_size = max(2, population_size // 5)
                    elite = [ind for ind, _ in fitness_scores[:elite_size]]
                    new_population = elite.copy()
                    while len(new_population) < population_size:
                        parent1 = random.choice(elite)
                        parent2 = random.choice(elite)
                        child = []
                        for i in range(top_k):
                            if random.random() < 0.5:
                                if parent1[i] not in child: child.append(parent1[i])
                            else:
                                if parent2[i] not in child: child.append(parent2[i])
                        while len(child) < top_k:
                            num = random.randint(0, 99)
                            if num not in child: child.append(num)
                        if random.random() < mutation_rate:
                            idx = random.randint(0, top_k - 1)
                            new_num = random.randint(0, 99)
                            while new_num in child: new_num = random.randint(0, 99)
                            child[idx] = new_num
                        new_population.append(child[:top_k])
                    population = new_population[:population_size]
                best_individual = max(population, key=lambda ind: self._evaluate_ga_fitness(ind, numbers, top_k))
                pred_numbers = best_individual[:top_k]
            
            boosted_predictions = self._apply_pattern_boost(numbers, pred_numbers)
            for rank, number in enumerate(boosted_predictions, 1):
                rows.append({
                    'date': target_date.strftime('%Y-%m-%d'), 'slot': slot, 'rank': rank,
                    'number': f"{number:02d}", 'source': 'scr10'
                })
        return pd.DataFrame(rows)
    
    def _evaluate_ga_fitness(self, individual, historical_numbers, top_k):
        if len(historical_numbers) < 20: return 0.5
        test_size = max(10, len(historical_numbers) // 5)
        test_data = historical_numbers[-test_size:]
        train_data = historical_numbers[:-test_size]
        if len(train_data) < 20: return 0.5
        hits = 0
        for i in range(1, len(test_data)):
            if test_data[i] in individual: hits += 1
        accuracy = hits / (len(test_data) - 1)
        unique_numbers = len(set(individual))
        diversity_bonus = unique_numbers / top_k
        pattern_bonus = 0
        for num in individual[:10]:
            if num in self.S40_numbers: pattern_bonus += 0.05
        return 0.6 * accuracy + 0.3 * diversity_bonus + 0.1 * min(pattern_bonus, 0.5)
    
    def _apply_pattern_boost(self, numbers, predictions):
        if not predictions: return predictions
        boosted = Counter()
        for num in predictions:
            score = 1.0
            if num in self.S40_numbers: score += 0.3
            for pattern_name, pattern_set in self.pattern_3digit.items():
                if num in pattern_set:
                    score += 0.2
                    break
            for pattern_name, pattern_set in self.pattern_4digit.items():
                if num in pattern_set:
                    score += 0.3
                    break
            for slot_name in self.SLOT_HOT:
                if num in self.SLOT_HOT[slot_name]: score += 0.4
            boosted[num] = score
        sorted_predictions = [num for num, _ in boosted.most_common(len(predictions))]
        return sorted_predictions[:len(predictions)]
    
    # ================== SCR11 - QUANTUM INSPIRED RANDOM FOREST ==================
    
    def generate_scr11_predictions(self, df, target_date, top_k=15):
        rows = []
        for slot in [1, 2, 3, 4]:
            slot_name = self.slot_names[slot]
            slot_data = df[df['slot'] == slot]
            
            if len(slot_data) < 50:
                pred_numbers = self._scr4_advanced_ensemble(slot_data['number'].tolist(), top_k)
            else:
                pred_numbers = self._scr11_quantum_rf_predict(slot_data, slot_name, top_k)
            
            for rank, number in enumerate(pred_numbers, 1):
                rows.append({
                    'date': target_date.strftime('%Y-%m-%d'), 'slot': slot, 'rank': rank,
                    'number': f"{number:02d}", 'source': 'scr11'
                })
        return pd.DataFrame(rows)
    
    def _scr11_quantum_rf_predict(self, slot_data, slot_name, top_k=15):
        try:
            numbers = slot_data['number'].tolist()
            dates = slot_data['date'].tolist()
            
            if len(numbers) < 50:
                return self._scr4_advanced_ensemble(numbers, top_k)
            
            X, y = [], []
            lookback = 20
            
            for i in range(lookback, len(numbers)):
                recent = numbers[i-lookback:i]
                features = []
                freq = Counter(recent)
                for n in range(100):
                    features.append(freq.get(n, 0) / lookback)
                
                positions = {}
                for idx, num in enumerate(recent):
                    if num not in positions: positions[num] = []
                    positions[num].append(idx)
                
                for n in range(100):
                    if n in positions and len(positions[n]) > 1:
                        last_pos = positions[n][-1]
                        features.append((lookback - 1 - last_pos) / lookback)
                    else:
                        features.append(1.0)
                
                pattern_score = 0
                for pattern_set in self.pattern_3digit.values():
                    if numbers[i-1] in pattern_set:
                        pattern_score += 1
                features.append(pattern_score / 10)
                
                hot_score = 1 if numbers[i-1] in self.SLOT_HOT[slot_name] else 0
                cold_score = 1 if numbers[i-1] in self.SLOT_COLD[slot_name] else 0
                features.extend([hot_score, cold_score])
                
                s40_score = 1 if numbers[i-1] in self.S40_numbers else 0
                features.append(s40_score)
                
                dow = dates[i].dayofweek
                features.append(dow / 6.0)
                
                X.append(features)
                y.append(numbers[i])
            
            if len(X) < 30:
                return self._scr4_advanced_ensemble(numbers, top_k)
            
            if slot_name not in self.rf_models:
                self.rf_models[slot_name] = RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
                )
            
            rf_model = self.rf_models[slot_name]
            rf_model.fit(X, y)
            
            pred_features = self._create_prediction_features(numbers, slot_name)
            probs = rf_model.predict_proba([pred_features])[0]
            all_numbers = list(range(100))
            if len(rf_model.classes_) < 90:
                return self._scr4_advanced_ensemble(numbers, top_k)
            
            prob_dict = {num: prob for num, prob in zip(rf_model.classes_, probs)}
            for num in all_numbers:
                if num not in prob_dict:
                    prob_dict[num] = 0.001
            
            sorted_numbers = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            pred_numbers = [num for num, _ in sorted_numbers[:top_k]]
            return pred_numbers
            
        except Exception as e:
            return self._scr4_advanced_ensemble(slot_data['number'].tolist(), top_k)
    
    def _create_prediction_features(self, numbers, slot_name):
        lookback = 20
        recent = numbers[-lookback:] if len(numbers) >= lookback else numbers
        
        features = []
        freq = Counter(recent)
        for n in range(100):
            features.append(freq.get(n, 0) / len(recent))
        
        positions = {}
        for idx, num in enumerate(recent):
            if num not in positions: positions[num] = []
            positions[num].append(idx)
        
        for n in range(100):
            if n in positions and len(positions[n]) > 1:
                last_pos = positions[n][-1]
                features.append((len(recent) - 1 - last_pos) / len(recent))
            else:
                features.append(1.0)
        
        pattern_score = 0
        for pattern_set in self.pattern_3digit.values():
            if numbers[-1] in pattern_set:
                pattern_score += 1
        features.append(pattern_score / 10)
        
        hot_score = 1 if numbers[-1] in self.SLOT_HOT[slot_name] else 0
        cold_score = 1 if numbers[-1] in self.SLOT_COLD[slot_name] else 0
        features.extend([hot_score, cold_score])
        
        s40_score = 1 if numbers[-1] in self.S40_numbers else 0
        features.append(s40_score)
        
        tomorrow_dow = (datetime.now().weekday() + 1) % 7
        features.append(tomorrow_dow / 6.0)
        
        return features
    
    # ================== SCR12 - GRADIENT BOOSTING + MLP ENSEMBLE ==================
    
    def generate_scr12_predictions(self, df, target_date, top_k=15):
        rows = []
        for slot in [1, 2, 3, 4]:
            slot_name = self.slot_names[slot]
            slot_data = df[df['slot'] == slot]
            
            if len(slot_data) < 60:
                pred_numbers = self._scr4_advanced_ensemble(slot_data['number'].tolist(), top_k)
            else:
                pred_numbers = self._scr12_gb_mlp_predict(slot_data, slot_name, top_k)
            
            for rank, number in enumerate(pred_numbers, 1):
                rows.append({
                    'date': target_date.strftime('%Y-%m-%d'), 'slot': slot, 'rank': rank,
                    'number': f"{number:02d}", 'source': 'scr12'
                })
        return pd.DataFrame(rows)
    
    def _scr12_gb_mlp_predict(self, slot_data, slot_name, top_k=15):
        try:
            numbers = slot_data['number'].tolist()
            dates = slot_data['date'].tolist()
            
            if len(numbers) < 60:
                return self._scr4_advanced_ensemble(numbers, top_k)
            
            X, y = [], []
            lookback = 25
            
            for i in range(lookback, len(numbers)):
                features = self._scr12_create_features(numbers[:i], slot_name, dates[i] if i < len(dates) else dates[-1])
                X.append(features)
                y.append(numbers[i])
            
            if len(X) < 40:
                return self._scr4_advanced_ensemble(numbers, top_k)
            
            if slot_name not in self.gb_models:
                self.gb_models[slot_name] = GradientBoostingClassifier(
                    n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42
                )
            
            if slot_name not in self.mlp_models:
                self.mlp_models[slot_name] = MLPClassifier(
                    hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True
                )
            
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            gb_model = self.gb_models[slot_name]
            mlp_model = self.mlp_models[slot_name]
            
            gb_model.fit(X_train, y_train)
            mlp_model.fit(X_train, y_train)
            
            gb_acc = gb_model.score(X_val, y_val)
            mlp_acc = mlp_model.score(X_val, y_val)
            total_acc = gb_acc + mlp_acc
            
            if total_acc == 0:
                gb_weight, mlp_weight = 0.5, 0.5
            else:
                gb_weight = gb_acc / total_acc
                mlp_weight = mlp_acc / total_acc
            
            recent_features = self._scr12_create_features(numbers, slot_name, datetime.now())
            
            gb_next_probs = gb_model.predict_proba([recent_features])[0]
            mlp_next_probs = mlp_model.predict_proba([recent_features])[0]
            
            all_numbers = list(range(100))
            final_probs = {}
            
            for num in all_numbers:
                if num in gb_model.classes_:
                    idx = list(gb_model.classes_).index(num)
                    gb_prob = gb_next_probs[idx]
                else:
                    gb_prob = 0.001
                
                if num in mlp_model.classes_:
                    idx = list(mlp_model.classes_).index(num)
                    mlp_prob = mlp_next_probs[idx]
                else:
                    mlp_prob = 0.001
                
                final_probs[num] = (gb_prob * gb_weight) + (mlp_prob * mlp_weight)
            
            for num in all_numbers:
                if num in self.SLOT_HOT[slot_name]:
                    final_probs[num] *= 1.2
                if num in self.S40_numbers:
                    final_probs[num] *= 1.1
            
            sorted_numbers = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)
            pred_numbers = [num for num, _ in sorted_numbers[:top_k]]
            
            return pred_numbers
            
        except Exception as e:
            return self._scr4_advanced_ensemble(slot_data['number'].tolist(), top_k)
    
    def _scr12_create_features(self, numbers, slot_name, current_date):
        features = []
        lookback = min(25, len(numbers))
        recent = numbers[-lookback:] if len(numbers) >= lookback else numbers
        
        freq = Counter(recent)
        weights = np.exp(np.linspace(0, 1, len(recent)))
        weights = weights / weights.sum()
        weighted_freq = {}
        for i, num in enumerate(recent):
            weighted_freq[num] = weighted_freq.get(num, 0) + weights[i]
        
        for n in range(100):
            features.append(weighted_freq.get(n, 0))
        
        positions = {}
        for idx, num in enumerate(numbers):
            if num not in positions: positions[num] = []
            positions[num].append(idx)
        
        for n in range(100):
            if n in positions and len(positions[n]) > 2:
                gaps = [positions[n][i] - positions[n][i-1] for i in range(1, len(positions[n]))]
                avg_gap = np.mean(gaps)
                std_gap = np.std(gaps) if len(gaps) > 1 else 1
                current_gap = len(numbers) - positions[n][-1]
                normalized_gap = (current_gap - avg_gap) / (std_gap + 1)
                features.append(min(max(normalized_gap, -3), 3) / 3)
            else:
                features.append(1.0)
        
        pattern_strength = 0
        for pattern_set in self.pattern_4digit.values():
            if numbers[-1] in pattern_set:
                pattern_strength += 1
        features.append(pattern_strength / 5)
        
        hot_momentum = 0
        cold_momentum = 0
        for num in recent[-5:]:
            if num in self.SLOT_HOT[slot_name]:
                hot_momentum += 1
            if num in self.SLOT_COLD[slot_name]:
                cold_momentum += 1
        features.extend([hot_momentum / 5, cold_momentum / 5])
        
        s40_count = sum(1 for num in recent if num in self.S40_numbers)
        features.append(s40_count / len(recent))
        
        if isinstance(current_date, (datetime, pd.Timestamp)):
            dow = current_date.dayofweek
            day_of_month = current_date.day
            month = current_date.month
        else:
            dow = 0
            day_of_month = 15
            month = 6
        
        features.extend([dow / 6.0, (day_of_month - 1) / 30.0, (month - 1) / 11.0])
        
        if len(recent) >= 3:
            changes = [abs(recent[i] - recent[i-1]) for i in range(1, len(recent))]
            volatility = np.std(changes) if len(changes) > 1 else 0
            features.append(min(volatility / 50, 1))
        else:
            features.append(0.5)
        
        if len(recent) >= 3:
            same_count = 1
            for i in range(1, min(5, len(recent))):
                if recent[-i] == recent[-i-1]:
                    same_count += 1
                else:
                    break
            features.append(same_count / 5)
        else:
            features.append(0.2)
        
        return features
    
    # ================== SCR13 - XGBOOST + ATTENTION MECHANISM ==================
    
    def generate_scr13_predictions(self, df, target_date, top_k=15):
        """NEW: XGBoost with Attention to Recent Patterns"""
        rows = []
        for slot in [1, 2, 3, 4]:
            slot_name = self.slot_names[slot]
            slot_data = df[df['slot'] == slot]
            
            if len(slot_data) < 80:
                pred_numbers = self._scr4_advanced_ensemble(slot_data['number'].tolist(), top_k)
            else:
                pred_numbers = self._scr13_xgboost_attention(slot_data, slot_name, top_k)
            
            for rank, number in enumerate(pred_numbers, 1):
                rows.append({
                    'date': target_date.strftime('%Y-%m-%d'), 'slot': slot, 'rank': rank,
                    'number': f"{number:02d}", 'source': 'scr13'
                })
        return pd.DataFrame(rows)
    
    def _scr13_xgboost_attention(self, slot_data, slot_name, top_k=15):
        try:
            numbers = slot_data['number'].tolist()
            
            if len(numbers) < 80:
                return self._scr4_advanced_ensemble(numbers, top_k)
            
            try:
                import xgboost as xgb
            except ImportError:
                return self._scr4_advanced_ensemble(numbers, top_k)
            
            X, y = [], []
            lookback = 30
            attention_window = 10
            
            for i in range(lookback, len(numbers)):
                recent = numbers[i-lookback:i]
                attention_weights = np.exp(np.linspace(0, 1, len(recent)))
                attention_weights = attention_weights / attention_weights.sum()
                
                features = []
                weighted_freq = defaultdict(float)
                for idx, num in enumerate(recent):
                    weight = attention_weights[idx]
                    weighted_freq[num] += weight
                
                for n in range(100):
                    features.append(weighted_freq.get(n, 0))
                
                pattern_score = 0
                for j in range(max(0, len(recent)-attention_window), len(recent)):
                    for pattern_set in self.pattern_3digit.values():
                        if recent[j] in pattern_set:
                            pattern_score += (j - (len(recent)-attention_window) + 1)
                features.append(pattern_score / (attention_window * 10))
                
                hot_momentum = 0
                for j in range(max(0, len(recent)-attention_window), len(recent)):
                    if recent[j] in self.SLOT_HOT[slot_name]:
                        hot_momentum += (j - (len(recent)-attention_window) + 1)
                features.append(hot_momentum / sum(range(1, attention_window+1)))
                
                positions = {}
                for idx, num in enumerate(numbers[:i]):
                    if num not in positions: positions[num] = []
                    positions[num].append(idx)
                
                for n in range(100):
                    if n in positions and len(positions[n]) > 1:
                        gaps = [positions[n][j] - positions[n][j-1] for j in range(1, len(positions[n]))]
                        recent_gaps = gaps[-3:] if len(gaps) >= 3 else gaps
                        avg_gap = np.mean(recent_gaps) if recent_gaps else 1
                        current_gap = i - positions[n][-1]
                        features.append(min(current_gap / (avg_gap + 1), 3))
                    else:
                        features.append(3.0)
                
                X.append(features)
                y.append(numbers[i])
            
            if len(X) < 50:
                return self._scr4_advanced_ensemble(numbers, top_k)
            
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            params = {
                'max_depth': 6,
                'eta': 0.1,
                'objective': 'multi:softprob',
                'num_class': 100,
                'eval_metric': 'mlogloss',
                'seed': 42
            }
            
            watchlist = [(dtrain, 'train'), (dval, 'eval')]
            bst = xgb.train(params, dtrain, num_boost_round=100,
                           evals=watchlist, early_stopping_rounds=20, verbose_eval=False)
            
            recent_features = self._scr13_create_features(numbers, slot_name, attention_window)
            dtest = xgb.DMatrix([recent_features])
            probs = bst.predict(dtest)[0]
            
            prob_dict = {n: probs[n] for n in range(100)}
            
            for n in range(100):
                if n in self.SLOT_HOT[slot_name]:
                    prob_dict[n] *= 1.2
                if n in self.S40_numbers:
                    prob_dict[n] *= 1.1
            
            sorted_numbers = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            pred_numbers = [num for num, _ in sorted_numbers[:top_k]]
            
            return pred_numbers
            
        except Exception as e:
            return self._scr4_advanced_ensemble(slot_data['number'].tolist(), top_k)
    
    def _scr13_create_features(self, numbers, slot_name, attention_window=10):
        lookback = min(30, len(numbers))
        recent = numbers[-lookback:] if len(numbers) >= lookback else numbers
        
        features = []
        attention_weights = np.exp(np.linspace(0, 1, len(recent)))
        attention_weights = attention_weights / attention_weights.sum()
        
        weighted_freq = defaultdict(float)
        for idx, num in enumerate(recent):
            weight = attention_weights[idx]
            weighted_freq[num] += weight
        
        for n in range(100):
            features.append(weighted_freq.get(n, 0))
        
        pattern_score = 0
        for j in range(max(0, len(recent)-attention_window), len(recent)):
            for pattern_set in self.pattern_3digit.values():
                if recent[j] in pattern_set:
                    pattern_score += (j - (len(recent)-attention_window) + 1)
        features.append(pattern_score / (attention_window * 10))
        
        hot_momentum = 0
        for j in range(max(0, len(recent)-attention_window), len(recent)):
            if recent[j] in self.SLOT_HOT[slot_name]:
                hot_momentum += (j - (len(recent)-attention_window) + 1)
        features.append(hot_momentum / sum(range(1, attention_window+1)))
        
        positions = {}
        for idx, num in enumerate(numbers):
            if num not in positions: positions[num] = []
            positions[num].append(idx)
        
        for n in range(100):
            if n in positions and len(positions[n]) > 1:
                gaps = [positions[n][j] - positions[n][j-1] for j in range(1, len(positions[n]))]
                recent_gaps = gaps[-3:] if len(gaps) >= 3 else gaps
                avg_gap = np.mean(recent_gaps) if recent_gaps else 1
                current_gap = len(numbers) - positions[n][-1]
                features.append(min(current_gap / (avg_gap + 1), 3))
            else:
                features.append(3.0)
        
        return features
    
    # ================== SCR14 - LIGHTGBM + TRANSFORMER ENSEMBLE ==================
    
    def generate_scr14_predictions(self, df, target_date, top_k=15):
        """NEW: LightGBM with Transformer-style embeddings"""
        rows = []
        for slot in [1, 2, 3, 4]:
            slot_name = self.slot_names[slot]
            slot_data = df[df['slot'] == slot]
            
            if len(slot_data) < 100:
                pred_numbers = self._scr13_xgboost_attention(slot_data, slot_name, top_k)
            else:
                pred_numbers = self._scr14_lightgbm_transformer(slot_data, slot_name, top_k)
            
            for rank, number in enumerate(pred_numbers, 1):
                rows.append({
                    'date': target_date.strftime('%Y-%m-%d'), 'slot': slot, 'rank': rank,
                    'number': f"{number:02d}", 'source': 'scr14'
                })
        return pd.DataFrame(rows)
    
    def _scr14_lightgbm_transformer(self, slot_data, slot_name, top_k=15):
        try:
            numbers = slot_data['number'].tolist()
            dates = slot_data['date'].tolist()
            
            if len(numbers) < 100:
                return self._scr13_xgboost_attention(slot_data, slot_name, top_k)
            
            try:
                import lightgbm as lgb
            except ImportError:
                return self._scr13_xgboost_attention(slot_data, slot_name, top_k)
            
            X, y = [], []
            lookback = 35
            
            for i in range(lookback, len(numbers)):
                features = self._scr14_create_transformer_features(numbers[:i], slot_name, dates[i])
                X.append(features)
                y.append(numbers[i])
            
            if len(X) < 60:
                return self._scr13_xgboost_attention(slot_data, slot_name, top_k)
            
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            params = {
                'objective': 'multiclass',
                'num_class': 100,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 127,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'seed': 42
            }
            
            gbm = lgb.train(params, train_data, num_boost_round=200,
                           valid_sets=[val_data], callbacks=[lgb.early_stopping(30)])
            
            recent_features = self._scr14_create_transformer_features(numbers, slot_name, target_date)
            probs = gbm.predict([recent_features], num_iteration=gbm.best_iteration)[0]
            
            prob_dict = {n: probs[n] for n in range(100)}
            
            transformer_boost = self._scr14_transformer_attention(numbers[-50:], slot_name)
            for n in range(100):
                if n in transformer_boost:
                    prob_dict[n] *= transformer_boost[n]
                if n in self.SLOT_HOT[slot_name]:
                    prob_dict[n] *= 1.15
                if n in self.S40_numbers:
                    prob_dict[n] *= 1.08
            
            sorted_numbers = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            pred_numbers = [num for num, _ in sorted_numbers[:top_k]]
            
            return pred_numbers
            
        except Exception as e:
            return self._scr13_xgboost_attention(slot_data, slot_name, top_k)
    
    def _scr14_create_transformer_features(self, numbers, slot_name, current_date):
        lookback = min(35, len(numbers))
        recent = numbers[-lookback:] if len(numbers) >= lookback else numbers
        
        features = []
        
        positional_encoding = np.sin(np.arange(len(recent)) / 10000 ** (np.arange(len(recent)) / len(recent)))
        positional_weights = np.exp(positional_encoding - np.max(positional_encoding))
        positional_weights = positional_weights / positional_weights.sum()
        
        positional_freq = defaultdict(float)
        for idx, num in enumerate(recent):
            weight = positional_weights[idx]
            positional_freq[num] += weight
        
        for n in range(100):
            features.append(positional_freq.get(n, 0))
        
        transformer_attention = self._scr14_self_attention(recent[-20:])
        for n in range(100):
            features.append(transformer_attention.get(n, 0))
        
        multi_head_attention = []
        for head in range(3):
            head_weights = np.sin((np.arange(len(recent)) + head) / 10000 ** (np.arange(len(recent)) / len(recent)))
            head_weights = np.exp(head_weights - np.max(head_weights))
            head_weights = head_weights / head_weights.sum()
            
            head_freq = defaultdict(float)
            for idx, num in enumerate(recent):
                weight = head_weights[idx]
                head_freq[num] += weight
            
            for n in range(100):
                features.append(head_freq.get(n, 0))
        
        residual_connections = []
        for window in [5, 10, 15]:
            window_data = recent[-window:] if len(recent) >= window else recent
            window_freq = Counter(window_data)
            for n in range(100):
                features.append(window_freq.get(n, 0) / len(window_data))
        
        layer_norm = np.std(recent) if len(recent) > 1 else 0
        features.append(min(layer_norm / 50, 1))
        
        dropout_mask = np.random.binomial(1, 0.9, 100)
        for mask_val in dropout_mask:
            features.append(mask_val)
        
        if isinstance(current_date, (datetime, pd.Timestamp)):
            time_embedding = np.sin(2 * np.pi * current_date.dayofweek / 7)
            features.append(time_embedding)
        else:
            features.append(0)
        
        return features
    
    def _scr14_self_attention(self, sequence):
        if not sequence:
            return {}
        
        attention_scores = {}
        for i, num_i in enumerate(sequence):
            for j, num_j in enumerate(sequence):
                if i != j:
                    similarity = 1.0 / (abs(num_i - num_j) + 1)
                    attention_scores[num_j] = attention_scores.get(num_j, 0) + similarity
        
        max_score = max(attention_scores.values()) if attention_scores else 1
        if max_score > 0:
            attention_scores = {k: v / max_score for k, v in attention_scores.items()}
        
        return attention_scores
    
    def _scr14_transformer_attention(self, sequence, slot_name):
        if len(sequence) < 10:
            return {}
        
        attention = {}
        for i in range(len(sequence)):
            query = sequence[i]
            key_scores = {}
            
            for j in range(len(sequence)):
                if i != j:
                    key = sequence[j]
                    score = 1.0 / (abs(query - key) + 1)
                    if key in self.SLOT_HOT[slot_name]:
                        score *= 1.5
                    if key in self.S40_numbers:
                        score *= 1.3
                    key_scores[key] = score
            
            total_score = sum(key_scores.values())
            if total_score > 0:
                for key, score in key_scores.items():
                    attention[key] = attention.get(key, 0) + (score / total_score)
        
        max_attn = max(attention.values()) if attention else 1
        if max_attn > 0:
            attention = {k: v / max_attn for k, v in attention.items()}
        
        return attention
    
    # ================== WEIGHT MANAGEMENT ==================
    
    def calculate_script_score(self, script_name, slot_name, predicted_numbers, actual_result):
        if actual_result is None: return 0.0
        if isinstance(actual_result, str):
            try: actual_result = int(actual_result)
            except: return 0.0
        max_rank = min(len(predicted_numbers), 15)

        for rank, num in enumerate(predicted_numbers[:max_rank], start=1):
            if num == actual_result: return (max_rank - rank + 1) / max_rank

        return 0.0
    
    def calculate_script_score_with_rank(self, script_name, slot_name, predicted_numbers, actual_result):
        if actual_result is None:
            return 0.0, 0
        try:
            actual_int = int(actual_result)
        except Exception:
            return 0.0, 0
    
        for rank, num in enumerate(predicted_numbers, start=1):
            try:
                if int(num) == actual_int:
                    n = len(predicted_numbers) if predicted_numbers else 1
                    score = (n - rank + 1) / n
                    return score, rank
            except Exception:
                continue
        return 0.0, 0

    def process_historical_predictions_for_weights(self):
        try:
            actual_df = pd.read_excel('number prediction learn.xlsx')
            if 'DATE' in actual_df.columns:
                actual_df['date'] = pd.to_datetime(actual_df['DATE'])
            else:
                actual_df['date'] = pd.to_datetime(actual_df['date'])
        except Exception as e:
            quiet_print(f"Error loading actual results: {e}")
            return
        
        prediction_files = list(self.COMBINED_DIR.glob("ultimate_predictions_*.xlsx"))
        if not prediction_files: 
            quiet_print("No prediction files found for weight update")
            return
        
        prediction_files.sort()
        
        scripts = ['scr1', 'scr2', 'scr3', 'scr4', 'scr6', 'scr7', 'scr8', 'scr9', 'scr10', 'scr11', 'scr12', 'scr13', 'scr14']
        for script in scripts:
            for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                model_id = f"{script}_{slot}"
                if model_id in self.model_weights:
                    self.model_weights[model_id]['performance_history'] = []
                    self.model_weights[model_id]['total_hits'] = 0
                    self.model_weights[model_id]['total_attempts'] = 0
        
        for pred_file in prediction_files:
            date_str = pred_file.stem.replace("ultimate_predictions_", "")
            
            try:
                try:
                    pred_df = pd.read_excel(pred_file, sheet_name='Predictions_Detailed_All')
                except:
                    pred_df = pd.read_excel(pred_file, sheet_name='Predictions_Detailed')
                if 'source' not in pred_df.columns: continue
                date_obj = pd.to_datetime(date_str)
                actual_for_date = actual_df[actual_df['date'].dt.date == date_obj.date()]
                if actual_for_date.empty: continue
                
                for slot_id, slot_name in self.slot_names.items():
                    actual_columns = [col for col in ['FRBD', 'GZBD', 'GALI', 'DSWR'] if col in actual_for_date.columns]
                    for actual_col in actual_columns:
                        if actual_col.upper() == slot_name.upper():
                            actual_result = actual_for_date[actual_col].iloc[0]
                            if pd.isna(actual_result) or str(actual_result).upper() == 'XX': continue
                            slot_preds = pred_df[pred_df['slot'] == slot_name]
                            if slot_preds.empty: slot_preds = pred_df[pred_df['slot'] == slot_id]
                            if slot_preds.empty: continue
                            
                            for script_name in scripts:
                                script_preds = slot_preds[slot_preds['source'] == script_name]
                                if not script_preds.empty:
                                    predicted_numbers = []
                                    for num in script_preds.sort_values('rank').head(15)['number'].tolist():
                                        try: predicted_numbers.append(int(num))
                                        except: continue
                                    score = self.calculate_script_score(script_name, slot_name, predicted_numbers, actual_result)
                                    model_id = f"{script_name}_{slot_name}"
                                    if model_id in self.model_weights:
                                        perf_entry = {'date': date_str, 'hit': score > 0, 'score': score}
                                        if 'performance_history' not in self.model_weights[model_id]:
                                            self.model_weights[model_id]['performance_history'] = []
                                        self.model_weights[model_id]['performance_history'].append(perf_entry)
                                        if len(self.model_weights[model_id]['performance_history']) > 30:
                                            self.model_weights[model_id]['performance_history'] = self.model_weights[model_id]['performance_history'][-30:]
                                        old_weight = self.model_weights[model_id]['weight']
                                        
                                        clamp_status = ""
                                        
                                        if score > 0:
                                            _score, hit_rank = self.calculate_script_score_with_rank(script_name, slot_name, predicted_numbers, actual_result)
                                            attempts = self.model_weights[model_id].get('total_attempts', 0)
                                            scale_factor = min(1.0, attempts / 10)
                                            
                                            if hit_rank == 1:
                                                rank_boost = 1.15 * scale_factor
                                            elif hit_rank <= 3:
                                                rank_boost = 1.10 * scale_factor
                                            elif hit_rank <= 5:
                                                rank_boost = 1.05 * scale_factor
                                            elif hit_rank <= 10:
                                                rank_boost = 1.02 * scale_factor
                                            else:
                                                rank_boost = 1.01 * scale_factor
                                            
                                            new_weight = old_weight * rank_boost
                                        else:
                                            attempts = self.model_weights[model_id].get('total_attempts', 0)
                                            hits = self.model_weights[model_id].get('total_hits', 0)
                                            
                                            if attempts < 5:
                                                penalty = 0.98
                                            elif attempts >= 10 and hits == 0:
                                                penalty = 0.7
                                            elif attempts >= 5 and hits == 0:
                                                penalty = 0.85
                                            else:
                                                penalty = 0.95
                                            
                                            new_weight = old_weight * penalty
                                        
                                        old_before_clamp = new_weight
                                        if new_weight < self.min_weight:
                                            new_weight = self.min_weight
                                            clamp_status = "CLAMP_MIN"
                                        elif new_weight > self.max_weight:
                                            new_weight = self.max_weight
                                            clamp_status = "CLAMP_MAX"
                                        
                                        self.model_weights[model_id]['clamp_status'] = clamp_status
                                        self.model_weights[model_id]['pre_clamp_weight'] = old_before_clamp
                                        
                                        if 'weight_history' not in self.model_weights[model_id]:
                                            self.model_weights[model_id]['weight_history'] = [old_weight]
                                        self.model_weights[model_id]['weight_history'].append(new_weight)
                                        self.model_weights[model_id]['weight'] = new_weight
                                        self.model_weights[model_id]['total_attempts'] = self.model_weights[model_id].get('total_attempts', 0) + 1
                                        if score > 0:
                                            self.model_weights[model_id]['total_hits'] = self.model_weights[model_id].get('total_hits', 0) + 1
                                        self.model_weights[model_id]['overall_accuracy'] = (
                                            self.model_weights[model_id].get('total_hits', 0) / 
                                            max(1, self.model_weights[model_id].get('total_attempts', 0))
                                        )
            except Exception as e:
                continue
        
        self.save_weights_to_json()
    
    def get_slot_roi_and_classification(self):
        try:
            if self.PNL_FILE.exists():
                per_slot_df = pd.read_excel(self.PNL_FILE, sheet_name='Per-Slot P&L')
                slot_stats = {}
                for slot_name in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                    slot_data = per_slot_df[per_slot_df['Slot'] == slot_name.lower()]
                    if not slot_data.empty:
                        total_stake = slot_data['Stake (₹)'].sum()
                        total_return = slot_data['Return (₹)'].sum()
                        if total_stake > 0:
                            roi = ((total_return - total_stake) / total_stake) * 100
                        else:
                            roi = 0.0
                        
                        if roi > 30:
                            classification = "superhero"
                        elif roi > 15:
                            classification = "hero"
                        elif roi < -50:
                            classification = "critical"
                        elif roi < -20:
                            classification = "weak"
                        else:
                            classification = "neutral"
                        
                        slot_stats[slot_name] = {'roi': roi, 'classification': classification}
                        
                        last_profits = slot_data.tail(3)['Profit (₹)'].tolist()
                        if len(last_profits) >= 2:
                            if last_profits[-1] > 0 and last_profits[-2] > 0:
                                self.slot_win_streak[slot_name] = min(5, self.slot_win_streak.get(slot_name, 0) + 1)
                                self.slot_loss_streak[slot_name] = 0
                            elif last_profits[-1] < 0 and last_profits[-2] < 0:
                                self.slot_loss_streak[slot_name] = min(5, self.slot_loss_streak.get(slot_name, 0) + 1)
                                self.slot_win_streak[slot_name] = 0
                    else:
                        slot_stats[slot_name] = {'roi': 0.0, 'classification': "neutral"}
                self.slot_roi = {slot: slot_stats[slot]['roi'] for slot in slot_stats}
                self.slot_classification = {slot: slot_stats[slot]['classification'] for slot in slot_stats}
                return slot_stats
            else:
                return {slot: {'roi': 0.0, 'classification': 'neutral'} for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']}
        except Exception as e:
            return {slot: {'roi': 0.0, 'classification': 'neutral'} for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']}
    
    def get_top_scripts(self, n=3):
        top_scripts = {}
        for slot_name in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
            slot_scripts = []
            scripts = ['scr1', 'scr2', 'scr3', 'scr4', 'scr6', 'scr7', 'scr8', 'scr9', 'scr10', 'scr11', 'scr12', 'scr13', 'scr14']
            for script in scripts:
                model_id = f"{script}_{slot_name}"
                if model_id in self.model_weights:
                    weight = self.model_weights[model_id]['weight']
                    accuracy = self.model_weights[model_id].get('overall_accuracy', 0.0)
                    attempts = self.model_weights[model_id].get('total_attempts', 0)
                    hits = self.model_weights[model_id].get('total_hits', 0)
                    clamp_status = self.model_weights[model_id].get('clamp_status', '')
                    slot_scripts.append({
                        'script': script, 'weight': weight, 'accuracy': accuracy,
                        'hits': hits, 'attempts': attempts, 'clamp_status': clamp_status
                    })
            slot_scripts.sort(key=lambda x: x['weight'], reverse=True)
            top_scripts[slot_name] = slot_scripts[:n]
        return top_scripts
    
    def weighted_merge_predictions(self, predictions_list, target_count=30):
        all_predictions = []
        for pred_df in predictions_list:
            for _, row in pred_df.iterrows():
                num = int(row['number'])
                rank = int(row['rank'])
                source = row['source']
                slot_id = row['slot']
                if slot_id in self.slot_names:
                    slot_name = self.slot_names[slot_id]
                else:
                    slot_name = slot_id
                model_id = f"{source}_{slot_name}"
                weight = self.model_weights.get(model_id, {}).get('weight', 0.3)
                top_k = 15
                rank_score = (top_k - rank + 1) / top_k
                score = weight * rank_score
                score += random.random() * 0.0001
                all_predictions.append({
                    'number': num, 'score': score, 'source': source,
                    'rank': rank, 'date': row['date'], 'slot': slot_id
                })
        merged_df = pd.DataFrame(all_predictions)
        grouped = merged_df.groupby(['date', 'slot', 'number'])['score'].sum().reset_index()
        grouped = grouped.sort_values(['date', 'slot', 'score'], ascending=[True, True, False])
        final_rows = []
        for (date, slot), group in grouped.groupby(['date', 'slot']):
            unique_numbers = []
            seen = set()
            for _, row in group.iterrows():
                num = row['number']
                if num not in seen and len(unique_numbers) < target_count:
                    seen.add(num)
                    unique_numbers.append(num)
            if len(unique_numbers) < target_count:
                remaining_numbers = [num for num in range(100) if num not in seen]
                unique_numbers.extend(remaining_numbers[:target_count - len(unique_numbers)])
            for rank, num in enumerate(unique_numbers[:target_count], 1):
                final_rows.append({
                    'date': date, 'slot': slot, 'rank': rank, 'number': f"{num:02d}"
                })
        return pd.DataFrame(final_rows)
    
    def apply_slot_bias(self, predictions_df):
        for slot_name in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
            slot_mask = predictions_df['slot'] == slot_name
            if not slot_mask.any(): continue
            slot_preds = predictions_df[slot_mask].copy()
            focus_range = self.slot_strategies[slot_name]['focus_range']
            if focus_range != range(0, 100):
                numbers_in_range = []
                numbers_out_range = []
                for _, row in slot_preds.iterrows():
                    try:
                        num = int(row['number'])
                        if num in focus_range:
                            numbers_in_range.append(row)
                        else:
                            numbers_out_range.append(row)
                    except:
                        numbers_out_range.append(row)
                if numbers_in_range:
                    predictions_df = predictions_df[~slot_mask]
                    combined = numbers_in_range + numbers_out_range[:max(0, 50 - len(numbers_in_range))]
                    predictions_df = pd.concat([predictions_df, pd.DataFrame(combined)], ignore_index=True)
        return predictions_df
    
    # ================== GZBD SMART PROTECTION ==================
    
    def get_gzbd_protection_strategy(self, roi, classification, win_streak, loss_streak):
        if classification == 'critical' or roi < -50:
            return {
                'numbers_k': 5,
                'stake_per_number': 3,
                'skip_digits': False,
                'max_k': 8,
                'reason': 'Critical losses - minimal betting'
            }
        elif classification == 'weak' or roi < -20:
            return {
                'numbers_k': 10,
                'stake_per_number': 5,
                'skip_digits': False,
                'max_k': 15,
                'reason': 'Weak performance - conservative'
            }
        elif loss_streak >= 3:
            return {
                'numbers_k': 8,
                'stake_per_number': 5,
                'skip_digits': False,
                'max_k': 12,
                'reason': f'Loss streak {loss_streak} - very conservative'
            }
        elif win_streak >= 3:
            return {
                'numbers_k': 18,
                'stake_per_number': 5,
                'skip_digits': False,
                'max_k': 22,
                'reason': f'Win streak {win_streak} - moderate boost'
            }
        else:
            return {
                'numbers_k': 15,
                'stake_per_number': 5,
                'skip_digits': False,
                'max_k': 20,
                'reason': 'Neutral - standard strategy'
            }
    
    # ================== DYNAMIC K ADJUSTMENT ==================
    
    def calculate_optimal_top_k_per_slot(self):
        try:
            if not self.PNL_FILE.exists():
                return {'FRBD': 18, 'GZBD': 15, 'GALI': 20, 'DSWR': 18}
            
            pnl_df = pd.read_excel(self.PNL_FILE, sheet_name='Per-Slot P&L')
            optimal_k = {}
            slot_stats = self.get_slot_roi_and_classification()

            MIN_DATA_DAYS = 5
            total_days = len(pnl_df['Date'].unique())

            if total_days < MIN_DATA_DAYS:
                return {'FRBD': 18, 'GZBD': 15, 'GALI': 20, 'DSWR': 18}
            
            for slot_name in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                slot_lower = slot_name.lower()
                slot_data = pnl_df[pnl_df['Slot'] == slot_lower]
                
                if len(slot_data) < 5:
                    classification = slot_stats[slot_name]['classification']
                    if classification == 'superhero':
                        optimal_k[slot_name] = 25
                    elif classification == 'hero':
                        optimal_k[slot_name] = 22
                    elif classification == 'critical':
                        optimal_k[slot_name] = 10
                    elif classification == 'weak':
                        optimal_k[slot_name] = 15
                    else:
                        optimal_k[slot_name] = 20
                    continue
                
                if slot_name == 'GZBD':
                    gzbd_strategy = self.get_gzbd_protection_strategy(
                        slot_stats[slot_name]['roi'],
                        slot_stats[slot_name]['classification'],
                        self.slot_win_streak.get(slot_name, 0),
                        self.slot_loss_streak.get(slot_name, 0)
                    )
                    optimal_k[slot_name] = gzbd_strategy['numbers_k']
                    continue
                
                best_k = 25
                best_profit = -float('inf')
                classification = slot_stats[slot_name]['classification']
                win_streak = self.slot_win_streak.get(slot_name, 0)
                loss_streak = self.slot_loss_streak.get(slot_name, 0)
                
                if classification == 'superhero' and win_streak >= 2:
                    k_range = [20, 22, 25, 28, 30]
                elif classification == 'critical' or loss_streak >= 2:
                    k_range = [10, 12, 15, 18, 20]
                elif classification == 'hero':
                    k_range = [18, 20, 22, 25, 28]
                else:
                    k_range = [15, 18, 20, 22, 25]
                
                for k in k_range:
                    total_stake = 0
                    total_return = 0
                    for _, row in slot_data.iterrows():
                        unit_stake = self.slot_strategies[slot_name]['unit_stake']
                        stake = k * unit_stake
                        total_stake += stake
                        actual = row['Actual']
                        top_numbers = row.get('Top Numbers', [])
                        if isinstance(top_numbers, str):
                            try: top_numbers = eval(top_numbers)
                            except: top_numbers = []
                        return_amount = 0
                        if actual in top_numbers[:k]: return_amount += 900
                        if row['Andar Hits'] == 1: return_amount += 90
                        if row['Bahar Hits'] == 1: return_amount += 90
                        total_return += return_amount
                    profit = total_return - total_stake
                    if profit > best_profit:
                        best_profit = profit
                        best_k = k
                
                optimal_k[slot_name] = best_k
            
            return optimal_k
            
        except Exception as e:
            return {'FRBD': 18, 'GZBD': 15, 'GALI': 20, 'DSWR': 18}
    
    # ================== WIN/LOSS STREAK MANAGEMENT ==================
    
    def apply_streak_adjustments(self, predictions_df):
        adjusted_df = predictions_df.copy()
        
        for slot_name in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
            slot_mask = adjusted_df['slot'] == slot_name
            if not slot_mask.any(): continue
            
            win_streak = self.slot_win_streak.get(slot_name, 0)
            loss_streak = self.slot_loss_streak.get(slot_name, 0)
            classification = self.slot_classification.get(slot_name, 'neutral')
            
            if win_streak >= 3 and classification in ['hero', 'superhero']:
                slot_indices = adjusted_df[slot_mask].index
                if len(slot_indices) > 10:
                    hot_nums = self.SLOT_HOT.get(slot_name, [])[:3]
                    for num in hot_nums:
                        if num not in adjusted_df.loc[slot_indices, 'number'].values:
                            new_row = {
                                'date': adjusted_df.loc[slot_indices[0], 'date'],
                                'slot': slot_name,
                                'rank': len(slot_indices) + 1,
                                'number': f"{num:02d}"
                            }
                            adjusted_df = pd.concat([adjusted_df, pd.DataFrame([new_row])], ignore_index=True)
            
            elif loss_streak >= 2 and classification in ['weak', 'critical']:
                slot_indices = adjusted_df[slot_mask].index
                s40_list = list(self.S40_numbers)
                if len(s40_list) > 0:
                    s40_sample = random.sample(s40_list, min(3, len(s40_list)))
                    for num in s40_sample:
                        if num not in adjusted_df.loc[slot_indices, 'number'].values:
                            new_row = {
                                'date': adjusted_df.loc[slot_indices[0], 'date'],
                                'slot': slot_name,
                                'rank': len(slot_indices) + 1,
                                'number': f"{num:02d}"
                            }
                            adjusted_df = pd.concat([adjusted_df, pd.DataFrame([new_row])], ignore_index=True)
        
        adjusted_df = adjusted_df.sort_values(['slot', 'rank']).reset_index(drop=True)
        adjusted_df['rank'] = adjusted_df.groupby('slot').cumcount() + 1
        
        return adjusted_df
    
    # ================== EMERGENCY BOOSTER (SAFE) ==================
    
    def apply_emergency_booster(self, predictions_df):
        try:
            if not self.PNL_FILE.exists(): return predictions_df
            
            pnl_df = pd.read_excel(self.PNL_FILE, sheet_name='Day Total P&L')
            if len(pnl_df) < 5: return predictions_df
            
            last_5_profits = pnl_df['Total Profit (₹)'].tail(5).sum()
            last_3_profits = pnl_df['Total Profit (₹)'].tail(3).sum()
            
            if last_5_profits < -2000 or last_3_profits < -1000:
                quiet_print("⚠️ SEVERE LOSSES - Applying SAFE booster")
                booster_applied = False
                
                for slot_name in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                    slot_mask = predictions_df['slot'] == slot_name
                    if not slot_mask.any(): continue
                    
                    slot_preds = predictions_df[slot_mask].copy()
                    current_nums = set()
                    for num_str in slot_preds['number']:
                        try: current_nums.add(int(num_str))
                        except: continue
                    
                    hot_nums = self.SLOT_HOT.get(slot_name, [])[:5]
                    added_count = 0
                    for num in hot_nums:
                        if num not in current_nums and added_count < 3:
                            new_rank = len(slot_preds) + 1 + added_count
                            new_row = pd.DataFrame([{
                                'date': slot_preds.iloc[0]['date'], 'slot': slot_name,
                                'rank': new_rank, 'number': f"{int(num):02d}"
                            }])
                            predictions_df = pd.concat([predictions_df, new_row], ignore_index=True)
                            current_nums.add(num)
                            added_count += 1
                            booster_applied = True
                    
                    if added_count < 5:
                        s40_added = 0
                        for s40_num in self.S40_numbers:
                            if s40_num not in current_nums and s40_added < 2:
                                new_rank = len(slot_preds) + 1 + added_count
                                new_row = pd.DataFrame([{
                                    'date': slot_preds.iloc[0]['date'], 'slot': slot_name,
                                    'rank': new_rank, 'number': f"{int(s40_num):02d}"
                                }])
                                predictions_df = pd.concat([predictions_df, new_row], ignore_index=True)
                                added_count += 1
                                s40_added += 1
                                booster_applied = True
                                if s40_added >= 2: break
                
                if booster_applied:
                    predictions_df = predictions_df.sort_values(['slot', 'rank']).reset_index(drop=True)
                    predictions_df['rank'] = predictions_df.groupby('slot').cumcount() + 1
                    quiet_print("✓ Safe booster applied (3 hot + 2 S40 numbers)")
            
            return predictions_df
            
        except Exception as e:
            return predictions_df
    
    # ================== DATA LEAKAGE PREVENTION ==================
    
    def validate_data_integrity(self, df):
        if df.empty:
            return True
        
        dates = df['date'].sort_values().tolist()
        for i in range(1, len(dates)):
            if dates[i] < dates[i-1]:
                quiet_print(f"⚠️ Date order issue: {dates[i-1]} -> {dates[i]}")
                return False
        
        date_diffs = []
        for i in range(1, len(dates)):
            diff = (dates[i] - dates[i-1]).days
            if diff > 7:
                date_diffs.append(diff)
        
        if date_diffs:
            quiet_print(f"⚠️ Found {len(date_diffs)} gaps >7 days")
        
        return True
    
    def update_patterns_from_data(self, df, target_date=None):
        if df.empty: return
        
        if target_date:
            if isinstance(target_date, (datetime, pd.Timestamp)):
                cutoff_date = target_date - timedelta(days=self.training_window)
                train_df = df[df['date'] < target_date]
            elif isinstance(target_date, date):
                target_datetime = datetime.combine(target_date, datetime.min.time())
                cutoff_date = target_datetime - timedelta(days=self.training_window)
                train_df = df[df['date'] < target_datetime]
            else:
                cutoff_date = df['date'].max() - timedelta(days=self.training_window)
                train_df = df.copy()
        else:
            cutoff_date = df['date'].max() - timedelta(days=self.training_window)
            train_df = df.copy()
        
        if isinstance(cutoff_date, date):
            cutoff_date = pd.Timestamp(cutoff_date)
        
        recent_df = train_df[train_df['date'] >= cutoff_date]
        if len(recent_df) < 30:
            recent_df = train_df.copy()
        
        if len(recent_df) < 10: return
        
        self.SLOT_HOT = {"FRBD": [], "GZBD": [], "GALI": [], "DSWR": []}
        self.SLOT_COLD = {"FRBD": [], "GZBD": [], "GALI": [], "DSWR": []}
        self.GLOBAL_MULTI_SLOT_HOT = []
        
        for slot_id, slot_name in self.slot_names.items():
            slot_data = recent_df[recent_df['slot'] == slot_id]
            if len(slot_data) < 5: continue
            numbers = slot_data['number'].tolist()
            freq = Counter(numbers)
            hot_count = min(10, len(freq) // 3)
            self.SLOT_HOT[slot_name] = [num for num, _ in freq.most_common(hot_count)]
            if len(freq) >= 10:
                cold_count = min(10, len(freq) // 3)
                self.SLOT_COLD[slot_name] = [num for num, _ in freq.most_common()[-cold_count:]]
        
        all_freq = Counter(recent_df['number'])
        hot_count = min(20, len(all_freq) // 2)
        self.GLOBAL_MULTI_SLOT_HOT = [num for num, _ in all_freq.most_common(hot_count)]
    
    def load_data(self, file_path):
        try:
            df = load_results_excel(file_path)
            df = self._ensure_long_format(df)
            
            if not self.validate_data_integrity(df):
                quiet_print("⚠️ Data integrity issues found")
            
            return df
        except Exception as e:
            quiet_print(f"Error loading data: {e}")
            raise
    
    def _ensure_long_format(self, df):
        if 'slot' in df.columns and 'number' in df.columns:
            return df[['date', 'slot', 'number']]
        
        wide_df = df.copy()
        if 'DATE' in wide_df.columns:
            wide_df['date'] = pd.to_datetime(wide_df['DATE'], errors='coerce')
        
        slot_cols = [c for c in ['FRBD', 'GZBD', 'GALI', 'DSWR'] if c in wide_df.columns]
        long_parts = []
        
        for col in slot_cols:
            part = wide_df[['date', col]].copy()
            part = part.rename(columns={col: 'number'})
            part['slot'] = self.slot_name_to_id.get(col, col)
            long_parts.append(part)
        
        long_df = pd.concat(long_parts, ignore_index=True)
        long_df['number'] = pd.to_numeric(long_df['number'], errors='coerce')
        long_df = long_df.dropna(subset=['date', 'slot', 'number'])
        long_df['slot'] = long_df['slot'].astype(int)
        long_df['number'] = long_df['number'].astype(int) % 100
        
        return long_df[['date', 'slot', 'number']]
    
    def get_prediction_date(self, df):
        last_date = df['date'].max().date()
        next_date = last_date + timedelta(days=1)
        
        try:
            actual_results = pd.read_excel('number prediction learn.xlsx')
            if 'DATE' in actual_results.columns:
                actual_dates = pd.to_datetime(actual_results['DATE']).dt.date.tolist()
                while next_date in actual_dates:
                    next_date += timedelta(days=1)
        except:
            pass
        
        last_day = calendar.monthrange(next_date.year, next_date.month)[1]
        if next_date.day == last_day:
            if next_date.month == 12:
                next_date = datetime(next_date.year + 1, 1, 1)
            else:
                next_date = datetime(next_date.year, next_date.month + 1, 1)
        else:
            next_date = datetime(next_date.year, next_date.month, next_date.day)
        
        return next_date
    
    # ================== MAIN PREDICTION GENERATION ==================
    
    def generate_all_predictions(self, df, predictions_per_slot=30):
        target_date = self.get_prediction_date(df)
        
        random_seed = int(target_date.strftime('%Y%m%d'))
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        self.update_patterns_from_data(df, target_date=target_date)
        
        self.process_historical_predictions_for_weights()
        
        self.optimal_top_k = self.calculate_optimal_top_k_per_slot()
        
        scr1_preds = self.generate_scr1_predictions(df, target_date, top_k=15)
        scr2_preds = self.generate_scr2_predictions(df, target_date, top_k=15)
        scr3_preds = self.generate_scr3_predictions(df, target_date, top_k=15)
        scr4_preds = self.generate_scr4_predictions(df, target_date, top_k=15)
        scr5_preds = self.generate_scr5_predictions(df, target_date, top_k=15)
        scr6_preds = self.generate_scr6_predictions(df, target_date, top_k=15)
        scr7_preds = self.generate_scr7_predictions(df, target_date, top_k=15)
        scr8_preds = self.generate_scr8_predictions(df, target_date, top_k=15)
        scr9_preds = self.generate_scr9_predictions(df, target_date, top_k=15)
        scr10_preds = self.generate_scr10_predictions(df, target_date, top_k=15)
        scr11_preds = self.generate_scr11_predictions(df, target_date, top_k=15)
        scr12_preds = self.generate_scr12_predictions(df, target_date, top_k=15)
        scr13_preds = self.generate_scr13_predictions(df, target_date, top_k=15)
        scr14_preds = self.generate_scr14_predictions(df, target_date, top_k=15)
        
        learning_signals = compute_learning_signals(df)
        all_preds = [scr1_preds, scr2_preds, scr3_preds, scr4_preds, scr5_preds, 
                    scr6_preds, scr7_preds, scr8_preds, scr9_preds, scr10_preds, 
                    scr11_preds, scr12_preds, scr13_preds, scr14_preds]
        
        for pred_df in all_preds:
            pred_df = apply_learning_to_dataframe(
                pred_df, learning_signals,
                slot_col='slot', number_col='number', rank_col='rank'
            )
        
        all_scripts_preds = pd.concat(all_preds, ignore_index=True)
        
        merged_predictions = self.weighted_merge_predictions(all_preds, target_count=predictions_per_slot)
        
        merged_predictions = apply_learning_to_dataframe(
            merged_predictions, learning_signals,
            slot_col='slot', number_col='number', rank_col='rank'
        )
        
        merged_predictions = self.apply_slot_bias(merged_predictions)
        merged_predictions = self.apply_streak_adjustments(merged_predictions)
        merged_predictions = self.apply_emergency_booster(merged_predictions)
        
        if merged_predictions['slot'].dtype in [np.int64, np.int32, int]:
            merged_predictions['slot'] = merged_predictions['slot'].apply(lambda x: self.slot_names[x])
        
        return merged_predictions, target_date, all_scripts_preds
    
    def create_output_file(self, predictions_df, prediction_date, all_scripts_preds=None):
        output_path = self.COMBINED_DIR / f"ultimate_predictions_{prediction_date.strftime('%Y-%m-%d')}.xlsx"
        if output_path.exists(): output_path.unlink()
        wide_df = predictions_df.pivot_table(
            index='date', columns='slot', values='number', aggfunc=lambda x: ', '.join(x)
        ).reset_index()
        column_order = ['date'] + [self.slot_names[i] for i in [1, 2, 3, 4]]
        wide_df = wide_df.reindex(columns=column_order)
        detailed_df = predictions_df.copy()
        detailed_df['score'] = detailed_df['rank'].apply(lambda x: 50 - x + 1)
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            wide_df.to_excel(writer, sheet_name='Predictions_Wide', index=False)
            detailed_df.to_excel(writer, sheet_name='Predictions_Detailed_Merged', index=False)
            if all_scripts_preds is not None:
                all_scripts_preds.to_excel(writer, sheet_name='Predictions_Detailed_All', index=False)
        return wide_df, output_path
    
    def get_weight_performance_tracker(self):
        tracker_data = {}
        scripts = ['scr1', 'scr2', 'scr3', 'scr4', 'scr6', 'scr7', 'scr8', 'scr9', 'scr10', 'scr11', 'scr12', 'scr13', 'scr14']
        for slot_name in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
            slot_tracker = []
            for script in scripts:
                model_id = f"{script}_{slot_name}"
                if model_id in self.model_weights:
                    weight_data = self.model_weights[model_id]
                    perf_history = weight_data.get('performance_history', [])
                    last_10_perf = perf_history[-10:] if len(perf_history) >= 10 else perf_history
                    hits_last_10 = sum(1 for perf in last_10_perf if perf.get('hit', False))
                    total_last_10 = len(last_10_perf)
                    hit_rate = (hits_last_10 / total_last_10 * 100) if total_last_10 > 0 else 0
                    weight_history = weight_data.get('weight_history', [0.3])
                    if len(weight_history) >= 2:
                        weight_change = weight_history[-1] - weight_history[-2]
                    else:
                        weight_change = 0
                    current_weight = weight_data.get('weight', 0.3)
                    clamp_status = weight_data.get('clamp_status', '')

                    if total_last_10 == 0:
                        status = 'NODATA'
                    else:
                        status = 'STRONG' if hit_rate >= 50 else 'WEAK' if hit_rate < 20 else 'NEUTRAL'

                    slot_tracker.append({
                        'script': script, 'hits': hits_last_10, 'total': total_last_10,
                        'hit_rate': hit_rate, 'weight': current_weight, 'weight_change': weight_change,
                        'status': status, 'clamp_status': clamp_status
                    })
            slot_tracker.sort(key=lambda x: x['weight'], reverse=True)
            tracker_data[slot_name] = slot_tracker
        return tracker_data
    
    def print_weight_performance_tracker(self):
        quiet_print("\n" + "="*45)
        quiet_print("WEIGHT TRACKER:")
        quiet_print("="*45)
        tracker_data = self.get_weight_performance_tracker()
        for slot_name in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
            slot_tracker = tracker_data[slot_name]
            total_scripts = len(slot_tracker)
            total_weight = sum(d.get('weight', 0.0) for d in slot_tracker)

            quiet_print(f"\n{slot_name} (Top 3 of {total_scripts}):")

            slot_tracker_top = slot_tracker[:3]
            for script_data in slot_tracker_top:
                script = script_data['script']
                hits = script_data['hits']
                total = script_data['total']
                hit_rate = script_data['hit_rate']
                weight = script_data['weight']
                weight_change = script_data['weight_change']
                status = script_data['status']
                clamp_status = script_data['clamp_status']
    
                if clamp_status == 'CLAMP_MIN':
                    weight_change_display = f"[MIN:0.100]"
                elif clamp_status == 'CLAMP_MAX':
                    weight_change_display = f"[MAX:0.800]"
                else:
                    if weight_change > 0:
                        weight_change_display = f"↑{weight_change:+.3f}"
                    elif weight_change < 0:
                        weight_change_display = f"↓{weight_change:+.3f}"
                    else:
                        weight_change_display = f"→{weight_change:+.3f}"
    
                if total == 0:
                    hit_display = "No data yet"
                else:
                    hit_display = f"{hits}/{total} ({hit_rate:.0f}%)"
    
                status_symbol = (
                    "➜" if status == 'NODATA'
                    else "✓" if status == 'STRONG'
                    else "⚠️" if status == 'WEAK'
                    else "➜"
                )
    
                quiet_print(f"  {status_symbol} {script}: {hit_display} W:{weight:.3f} {weight_change_display}")
        quiet_print("\n" + "-"*45)
    
    def print_pnl_verification(self, latest_day_breakdown):
        if not latest_day_breakdown: return
        quiet_print("\n" + "="*45)
        quiet_print("PNL VERIFICATION:")
        quiet_print("="*45)
        total_profit = 0
        for slot_data in latest_day_breakdown:
            slot_name = slot_data['Slot'].upper()
            actual = slot_data['Actual']
            direct_hit = slot_data['Direct Hits'] == 1
            andar_hit = slot_data['Andar Hits'] == 1
            bahar_hit = slot_data['Bahar Hits'] == 1
            stake = slot_data['Stake (₹)']
            return_amount = slot_data['Return (₹)']
            profit = slot_data['Profit (₹)']
            total_profit += profit
            
            unit_stake = self.slot_strategies[slot_name]['unit_stake']
            quiet_print(f"{slot_name} ({unit_stake}₹/num): Actual {actual} | "
                       f"{'✓' if direct_hit else '✗'} Dir | "
                       f"{'✓' if andar_hit else '✗'} And | "
                       f"{'✓' if bahar_hit else '✗'} Bah | "
                       f"Profit ₹{profit:+,.0f}")
        quiet_print(f"Total Profit: ₹{total_profit:+,.0f}")
    
    def print_clean_summary(self, predictions_df, target_date):
        quiet_print("\n" + "="*45)
        quiet_print(f"PREDICTIONS {target_date.date()}")
        quiet_print("="*45)
        slot_stats = self.get_slot_roi_and_classification()
        top_scripts = self.get_top_scripts(n=2)
        
        for slot_name in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
            slot_predictions = predictions_df[predictions_df['slot'] == slot_name]
            optimal_k = self.optimal_top_k.get(slot_name, 25)
            unit_stake = self.slot_strategies[slot_name]['unit_stake']
            numbers_list = slot_predictions.sort_values('rank')['number'].tolist()[:optimal_k]
            numbers_str = ", ".join([f"{int(num):02d}" for num in numbers_list])
            
            from pnl_helper import PNLCalculator
            calculator = PNLCalculator()
            top_numbers = []
            for num_str in numbers_list[:3]:
                try: top_numbers.append(int(num_str))
                except: continue
            andar_digits, bahar_digits = calculator.get_andar_bahar_candidates(top_numbers)
            
            classification = self.slot_classification.get(slot_name, "neutral")
            roi = self.slot_roi.get(slot_name, 0.0)
            win_streak = self.slot_win_streak.get(slot_name, 0)
            loss_streak = self.slot_loss_streak.get(slot_name, 0)
            
            stake = optimal_k * unit_stake
            
            streak_display = f"W{win_streak}" if win_streak > 0 else f"L{loss_streak}" if loss_streak > 0 else "-"
            
            quiet_print(f"{slot_name} ({classification})[{streak_display}] K:{optimal_k} ROI:{roi:+.1f}% Stk:₹{stake}({unit_stake}₹)")
            quiet_print(f"  Nums:{numbers_str}")
            quiet_print(f"  A:{sorted(andar_digits)} B:{sorted(bahar_digits)}")
            
            if slot_name in top_scripts and top_scripts[slot_name]:
                top_script = top_scripts[slot_name][0]
                quiet_print(f"  Top:{top_script['script']} W:{top_script['weight']:.3f}")
        
        self.print_weight_performance_tracker()
    
    def run_pnl_calculations(self):
        try:
            from pnl_helper import PNLCalculator
            calculator = PNLCalculator()
            calculator.optimal_top_k = self.optimal_top_k
        
            # ✅ DEBUG: Show what we're processing
            if not QUIET_MODE:
                import os
                pred_dir = calculator.PREDICTIONS_DIR
                if pred_dir.exists():
                    pred_files = list(pred_dir.glob("ultimate_predictions_*.xlsx"))
                    print(f"ℹ️ Found {len(pred_files)} prediction files")
        
            per_slot_data, day_total_data, cumulative_data, latest_day_breakdown = calculator.run(debug=False)
        
            # ✅ Check if PNL file was created/updated
            if calculator.PNL_FILE.exists():
                file_size = calculator.PNL_FILE.stat().st_size
                print(f"ℹ️ PNL file size: {file_size} bytes")
        
            if latest_day_breakdown:
                self.print_pnl_verification(latest_day_breakdown)
            elif calculator.PNL_FILE.exists():
                # Try to load and show last day's P&L
                try:
                    day_df = pd.read_excel(calculator.PNL_FILE, sheet_name='Day Total P&L')
                    if not day_df.empty:
                        print(f"\n📊 P&L Data Available: {len(day_df)} days")
                        # Show last 3 days
                        for idx, row in day_df.tail(3).iterrows():
                            profit = row['Total Profit (₹)']
                            roi = (profit / row['Total Stake (₹)'] * 100) if row['Total Stake (₹)'] > 0 else 0
                            color = "🟢" if profit >= 0 else "🔴"
                            print(f"{color} {row['Date']}: ₹{profit:+,.0f} ({roi:+.1f}%)")
                except:
                    pass
        
                if day_total_data:
                    quiet_print("\n" + "="*45)
                    quiet_print("DAILY P&L:")
                    quiet_print("="*45)
                    # ✅ FIX: Check if we have at least 5 days
                    display_days = day_total_data[-5:] if len(day_total_data) >= 5 else day_total_data
                    for day in display_days:
                        profit = day['Total Profit (₹)']
                        roi = (profit / day['Total Stake (₹)'] * 100) if day['Total Stake (₹)'] > 0 else 0
                        profit_color = "🟢" if profit >= 0 else "🔴"
                        quiet_print(f"{profit_color} {day['Date']}: ₹{profit:+,.0f} [{roi:+.1f}%]")
        
                if cumulative_data:
                    latest = cumulative_data[-1]
                    cumulative_profit = latest['Cumulative Profit (₹)']
                    cumulative_stake = latest['Cumulative Stake (₹)']
                    cumulative_roi = (cumulative_profit / cumulative_stake * 100) if cumulative_stake > 0 else 0
            
                    self.total_profit += cumulative_profit
                    self.total_days = len(cumulative_data)
                    self.save_performance_tracker()
            
                    quiet_print(f"\nCumulative: ₹{cumulative_profit:+,.0f} | ROI:{cumulative_roi:+.1f}% | Days:{self.total_days}")
        
                return per_slot_data, day_total_data, cumulative_data
        
        except Exception as e:
            quiet_print(f"ℹ️ P&L calculation pending - waiting for actual results")
            return [], [], []

def main():
    predictor = UltimateAllPredictor()
    file_path = 'number prediction learn.xlsx'
    quiet_print("Loading data...")
    df = predictor.load_data(file_path)
    if df is not None and len(df) > 0:
        predictions, prediction_date, all_scripts_preds = predictor.generate_all_predictions(df, predictions_per_slot=30)
        wide_predictions, output_path = predictor.create_output_file(predictions, prediction_date, all_scripts_preds)
        quiet_print(f"\n✓ File saved: {output_path.relative_to(predictor.PROJECT_DIR)}")
        predictor.print_clean_summary(predictions, prediction_date)
        predictor.run_pnl_calculations()
    else:
        quiet_print("Failed to load data.")

if __name__ == "__main__":
    main()