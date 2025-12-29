import pandas as pd
import json
from pathlib import Path
import numpy as np

import sys

QUIET_MODE = '--quiet' in sys.argv

_print = print

def quiet_print(*args, **kwargs):
    if not QUIET_MODE:
        _print(*args, **kwargs)


class EnhancedPerformanceTracker:
    def __init__(self):
        self.PROJECT_DIR = Path(__file__).resolve().parent
        self.WEIGHTS_FILE = self.PROJECT_DIR / "model_weights.json"
        self.PNL_FILE = self.PROJECT_DIR / "pnl.xlsx"
        self.OUTPUT_FILE = self.PROJECT_DIR / "features_performance.csv"
    
    def generate_summary(self):
        try:
            if not self.PNL_FILE.exists():
                quiet_print("ℹ️ P&L file not found. Run scr1-10.py and add actual results first.")
                return False

            pnl_df = pd.read_excel(self.PNL_FILE, sheet_name='Per-Slot P&L')
            day_total_df = pd.read_excel(self.PNL_FILE, sheet_name='Day Total P&L')

            if pnl_df.empty or day_total_df.empty:
                quiet_print("⚠️ P&L file empty or corrupted.")
                return False

            with open(self.WEIGHTS_FILE, 'r') as f:
                weights_data = json.load(f)
            summary = []
            
            # Recent days
            last_5_days = day_total_df.tail(5)
            for _, row in last_5_days.iterrows():
                summary.append({
                    'Type': 'DAILY', 'Date': str(row['Date']), 'Stake': int(row['Total Stake (₹)']),
                    'Return': int(row['Total Return (₹)']), 'Profit': int(row['Total Profit (₹)']),
                    'ROI': f"{row['Total Profit (₹)']/row['Total Stake (₹)']*100:.1f}%" if row['Total Stake (₹)'] > 0 else "0%"
                })
            
            # Top scripts
            weights = weights_data.get('weights', {})
            slots = ['FRBD', 'GZBD', 'GALI', 'DSWR']
            for slot in slots:
                slot_weights = []
                scripts = ['scr1', 'scr2', 'scr3', 'scr4', 'scr6', 'scr7', 'scr8', 'scr9', 'scr10', 'scr11', 'scr12', 'scr13', 'scr14']
                for script in scripts:
                    model_id = f"{script}_{slot}"
                    if model_id in weights:
                        w = weights[model_id].get('weight', 0.3)
                        hits = weights[model_id].get('total_hits', 0)
                        attempts = weights[model_id].get('total_attempts', 1)
                        acc = (hits / attempts * 100) if attempts > 0 else 0
                        clamp_status = weights[model_id].get('clamp_status', '')
                        slot_weights.append((script, w, acc, hits, attempts, clamp_status))
                slot_weights.sort(key=lambda x: x[1], reverse=True)
                for script, weight, acc, hits, attempts, clamp_status in slot_weights[:2]:
                    clamp_display = f"[{clamp_status}]" if clamp_status else ""
                    summary.append({
                        'Type': 'WEIGHT', 'Slot': slot, 'Script': script,
                        'Weight': f"{weight:.3f}{clamp_display}", 'Accuracy': f"{acc:.1f}%", 'Hits': f"{hits}/{attempts}"
                    })
            
            # Top-K analysis
            top_k_analysis = self.calculate_top_k_roi_analysis(pnl_df)
            summary.extend(top_k_analysis[:2])
            
            # Overall
            total_stake = day_total_df['Total Stake (₹)'].sum()
            total_return = day_total_df['Total Return (₹)'].sum()
            total_profit = total_return - total_stake
            overall_roi = (total_profit / total_stake * 100) if total_stake > 0 else 0
            
            summary.append({
                'Type': 'OVERALL', 'Metric': 'Total Profit',
                'Value': f"₹{total_profit:+,.0f}", 'ROI': f"{overall_roi:+.1f}%",
                'Days': len(day_total_df), 'Avg_Daily': f"₹{total_profit/len(day_total_df):+,.0f}" if len(day_total_df) > 0 else "₹0"
            })
            
            # Slot performance with unit stake info
            slot_performance = []
            unit_stakes = {'FRBD': 10, 'GZBD': 5, 'GALI': 10, 'DSWR': 10}
            for slot in slots:
                slot_lower = slot.lower()
                slot_data = pnl_df[pnl_df['Slot'] == slot_lower]
                if not slot_data.empty:
                    slot_stake = slot_data['Stake (₹)'].sum()
                    slot_return = slot_data['Return (₹)'].sum()
                    slot_profit = slot_return - slot_stake
                    slot_roi = (slot_profit / slot_stake * 100) if slot_stake > 0 else 0
                    unit_stake = unit_stakes.get(slot, 10)
                    slot_performance.append((slot, slot_roi, slot_profit, unit_stake))
            
            if slot_performance:
                best_slot = max(slot_performance, key=lambda x: x[1])
                worst_slot = min(slot_performance, key=lambda x: x[1])
                summary.append({
                    'Type': 'BEST_SLOT', 'Slot': best_slot[0],
                    'ROI': f"{best_slot[1]:+.1f}%", 'Profit': f"₹{best_slot[2]:+,.0f}", 'Unit_Stake': f"₹{best_slot[3]}"
                })
                summary.append({
                    'Type': 'WORST_SLOT', 'Slot': worst_slot[0],
                    'ROI': f"{worst_slot[1]:+.1f}%", 'Profit': f"₹{worst_slot[2]:+,.0f}", 'Unit_Stake': f"₹{worst_slot[3]}"
                })
            
            df = pd.DataFrame(summary)
            df.to_csv(self.OUTPUT_FILE, index=False)
            
            # COMPACT console output
            quiet_print("\n" + "="*35)
            quiet_print("PERFORMANCE SUMMARY")
            quiet_print("="*35)
            quiet_print(f"Overall: ₹{total_profit:+,.0f} ({overall_roi:+.1f}%) over {len(day_total_df)} days")
            
            quiet_print("\nRecent 5 Days:")
            for entry in summary[:5]:
                if entry['Type'] == 'DAILY':
                    profit = int(entry['Profit'])
                    profit_color = "🟢" if profit >= 0 else "🔴"
                    quiet_print(f"{profit_color} {entry['Date']}: ₹{profit} ({entry['ROI']})")
            
            if top_k_analysis:
                quiet_print(f"\nTop-K: K={top_k_analysis[0]['K']} ₹{top_k_analysis[0]['Profit']} ({top_k_analysis[0]['ROI']})")
            
            best_slot_info = summary[-2]
            worst_slot_info = summary[-1]
            quiet_print(f"\nBest: {best_slot_info['Slot']} {best_slot_info['ROI']}({best_slot_info['Unit_Stake']}₹) "
                       f"Worst: {worst_slot_info['Slot']} {worst_slot_info['ROI']}({worst_slot_info['Unit_Stake']}₹)")
            quiet_print(f"✓ Summary saved ({self.OUTPUT_FILE.stat().st_size} bytes)")
            return True
            
        except Exception as e:
            quiet_print(f"Error: {e}")
            return False
    
    def calculate_top_k_roi_analysis(self, pnl_df):
        analysis = []
        unit_stakes = {'FRBD': 10, 'GZBD': 5, 'GALI': 10, 'DSWR': 10}
        
        for k in [15, 20, 25, 30]:
            total_stake = 0
            total_return = 0
            for _, row in pnl_df.iterrows():
                slot_name = row['Slot'].upper()
                stake_per_number = unit_stakes.get(slot_name, 10)
                
                top_numbers = row.get('Top Numbers', [])
                if isinstance(top_numbers, str):
                    try: top_numbers = eval(top_numbers)
                    except: top_numbers = []
                
                andar_candidates, bahar_candidates = self.get_andar_bahar_candidates(top_numbers)
                
                numbers_stake = k * stake_per_number
                andar_stake = 10 * len(andar_candidates)
                bahar_stake = 10 * len(bahar_candidates)
                slot_stake = numbers_stake + andar_stake + bahar_stake
                
                total_stake += slot_stake
                actual = row['Actual']
                
                return_amount = 0
                if actual in top_numbers[:k]: return_amount += 900
                if row['Andar Hits'] == 1: return_amount += 90
                if row['Bahar Hits'] == 1: return_amount += 90
                total_return += return_amount
            profit = total_return - total_stake
            roi = (profit / total_stake * 100) if total_stake > 0 else 0
            analysis.append({
                'Type': 'TOP_K_ANALYSIS', 'K': k,
                'Profit': f"₹{profit:+,.0f}", 'ROI': f"{roi:+.1f}%"
            })
        analysis.sort(key=lambda x: float(x['ROI'].replace('%', '').replace('+', '').replace('-', '')), reverse=True)
        return analysis
    
    def get_andar_bahar_candidates(self, top_numbers):
        if not top_numbers or len(top_numbers) < 3:
            if top_numbers and len(top_numbers) > 0:
                first_num = top_numbers[0]
                tens = first_num // 10
                ones = first_num % 10
                return {tens}, {ones}
            else:
                return set(), set()
        top_3_numbers = []
        for num in top_numbers[:3]:
            try:
                top_3_numbers.append(int(num))
            except: continue
        if not top_3_numbers: return set(), set()
        tens_set = set()
        ones_set = set()
        for num in top_3_numbers:
            tens_set.add(num // 10)
            ones_set.add(num % 10)
        mirror_tens = set()
        mirror_ones = set()
        for num in top_3_numbers:
            tens = num // 10
            ones = num % 10
            mirror_tens.add(ones)
            mirror_ones.add(tens)
        all_tens = tens_set.union(mirror_tens)
        all_ones = ones_set.union(mirror_ones)
        return all_tens, all_ones

if __name__ == "__main__":
    tracker = EnhancedPerformanceTracker()
    tracker.generate_summary()