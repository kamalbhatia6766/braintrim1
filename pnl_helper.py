import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

class PNLCalculator:
    def __init__(self):
        self.PROJECT_DIR = Path(__file__).resolve().parent
        self.PREDICTIONS_DIR = self.PROJECT_DIR / "predictions" / "ultimate_all"
        self.PNL_FILE = self.PROJECT_DIR / "pnl.xlsx"
        self.PNL_HISTORY_FILE = self.PROJECT_DIR / "pnl_history.json"
        self.slot_names = {1: "FRBD", 2: "GZBD", 3: "GALI", 4: "DSWR"}
        self.slot_name_to_id = {name: sid for sid, name in self.slot_names.items()}
        self.optimal_top_k = None
        self.unit_stakes = {'FRBD': 10, 'GZBD': 5, 'GALI': 10, 'DSWR': 10}
        
        # Load existing P&L history to prevent recalculation
        self.load_pnl_history()
    
    def load_pnl_history(self):
        """Load existing P&L history to avoid recalculating past days"""
        self.pnl_history = defaultdict(dict)
        if self.PNL_HISTORY_FILE.exists():
            try:
                import json
                with open(self.PNL_HISTORY_FILE, 'r') as f:
                    self.pnl_history = json.load(f)
                # Convert date keys back to proper format
                self.pnl_history = {k: v for k, v in self.pnl_history.items()}
                
                # ✅ CRITICAL FIX: Remove entries without profit data
                dates_to_remove = []
                for date_str, slot_data in self.pnl_history.items():
                    for slot_name, data in slot_data.items():
                        if 'profit' not in data or data.get('profit', None) is None:
                            dates_to_remove.append(date_str)
                            break
                
                for date_str in set(dates_to_remove):
                    if date_str in self.pnl_history:
                        del self.pnl_history[date_str]
                        
            except:
                self.pnl_history = defaultdict(dict)
    
    def save_pnl_history(self):
        """Save P&L history to prevent recalculation"""
        try:
            import json
            with open(self.PNL_HISTORY_FILE, 'w') as f:
                json.dump(self.pnl_history, f, indent=2)
        except:
            pass
    
    def load_actual_results(self):
        try:
            df = pd.read_excel('number prediction learn.xlsx')
            if 'DATE' in df.columns:
                df['date'] = pd.to_datetime(df['DATE'])
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            else:
                return pd.DataFrame()
            result_cols = ['date', 'FRBD', 'GZBD', 'GALI', 'DSWR']
            available_cols = [col for col in result_cols if col in df.columns]
            if len(available_cols) < 2: return pd.DataFrame()
            return df[available_cols]
        except Exception as e:
            return pd.DataFrame()
    
    def extract_date_from_filename(self, filename):
        try:
            date_str = filename.stem.replace("ultimate_predictions_", "")
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except:
            return None
    
    def load_predictions_file(self, file_path):
        try:
            try:
                predictions_df = pd.read_excel(file_path, sheet_name='Predictions_Detailed_Merged')
            except:
                predictions_df = pd.read_excel(file_path, sheet_name='Predictions_Detailed')
            if predictions_df['slot'].dtype in [np.int64, np.int32, int]:
                predictions_df['slot'] = predictions_df['slot'].apply(
                    lambda x: self.slot_names.get(x, str(x))
                )
            return predictions_df
        except Exception as e:
            return None
    
    def get_andar_bahar_candidates(self, top_numbers):
        if not top_numbers or len(top_numbers) < 3:
            if top_numbers and len(top_numbers) > 0:
                first_num = top_numbers[0]
                if isinstance(first_num, str):
                    try: first_num = int(first_num)
                    except: return set(), set()
                tens = first_num // 10
                ones = first_num % 10
                return {tens}, {ones}
            else:
                return set(), set()
        top_3_numbers = []
        for num in top_numbers[:3]:
            try:
                if isinstance(num, str): num = int(num)
                top_3_numbers.append(num)
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
    
    def calculate_pnl_for_slot(self, date_str, slot_name, actual_number, predictions_df, original_k=None):
        """Calculate P&L using ORIGINAL K values (stored in history)"""
        slot_preds = predictions_df[predictions_df['slot'] == slot_name]
        if slot_preds.empty: return None
        
        # Use original K if available in history, otherwise current optimal
        if date_str in self.pnl_history and slot_name in self.pnl_history[date_str]:
            k_used = self.pnl_history[date_str][slot_name].get('k_used', 25)
        else:
            k_used = original_k if original_k else 25
        
        if slot_name.upper() == 'GZBD':
            stake_per_number_eff = 5
        else:
            stake_per_number_eff = 10
        
        stake_per_digit_eff = 10
        
        pick_n = k_used
        top_numbers = []
        for num in slot_preds.sort_values('rank').head(pick_n)['number'].tolist():
            try:
                if isinstance(num, str): num = int(num)
                top_numbers.append(int(num))
            except: continue
        
        andar_candidates, bahar_candidates = self.get_andar_bahar_candidates(top_numbers)
        if pd.isna(actual_number) or str(actual_number).upper() == 'XX': return None
        try: actual_int = int(actual_number)
        except: return None
        
        direct_hit = actual_int in top_numbers
        actual_tens = actual_int // 10
        actual_ones = actual_int % 10
        andar_hit = actual_tens in andar_candidates
        bahar_hit = actual_ones in bahar_candidates
        
        numbers_stake = k_used * stake_per_number_eff
        andar_stake = stake_per_digit_eff * len(andar_candidates)
        bahar_stake = stake_per_digit_eff * len(bahar_candidates)
        total_stake = numbers_stake + andar_stake + bahar_stake
        
        return_amount = 0
        if direct_hit: return_amount += 900
        if andar_hit: return_amount += 90
        if bahar_hit: return_amount += 90
        
        profit = return_amount - total_stake
        
        # ✅ CRITICAL FIX: Only store in history if we have actual result
        if actual_number is not None and str(actual_number).upper() != 'XX':
            if date_str not in self.pnl_history:
                self.pnl_history[date_str] = {}
            self.pnl_history[date_str][slot_name] = {
                'k_used': k_used,
                'stake_per_number': stake_per_number_eff,
                'top_numbers': top_numbers,
                'andar_candidates': sorted(andar_candidates),
                'bahar_candidates': sorted(bahar_candidates),
                'actual': actual_int,
                'profit': profit
            }
        
        return {
            'Date': date_str, 'Slot': slot_name.lower(), 'Stake (₹)': total_stake,
            'Numbers Stake (₹)': numbers_stake, 'Andar Stake (₹)': andar_stake, 'Bahar Stake (₹)': bahar_stake,
            'Total Numbers': k_used, 'Actual': actual_int,
            'Direct Hits': 1 if direct_hit else 0, 'Andar Hits': 1 if andar_hit else 0,
            'Bahar Hits': 1 if bahar_hit else 0, 'Return (₹)': return_amount,
            'Profit (₹)': profit, 'Top Numbers': top_numbers,
            'Andar Candidates': sorted(andar_candidates), 'Bahar Candidates': sorted(bahar_candidates),
            'Actual Tens': actual_tens, 'Actual Ones': actual_ones, 'Optimal K': k_used
        }
    
    def process_all_predictions(self):
        """Process predictions - ONLY calculate for NEW days, preserve past days"""
        actual_results = self.load_actual_results()
        if actual_results.empty: 
            print("ℹ️ No actual results found in Excel file")
            return [], [], [], []
        prediction_files = list(self.PREDICTIONS_DIR.glob("ultimate_predictions_*.xlsx"))
        if not prediction_files: 
            print("ℹ️ No prediction files found")
            return [], [], [], []
        prediction_files.sort()
        
        # Load existing P&L data to preserve
        existing_per_slot_data = []
        existing_day_totals = []
        existing_cumulative = []
        
        if self.PNL_FILE.exists():
            try:
                existing_per_slot_df = pd.read_excel(self.PNL_FILE, sheet_name='Per-Slot P&L')
                existing_day_total_df = pd.read_excel(self.PNL_FILE, sheet_name='Day Total P&L')
                existing_cumulative_df = pd.read_excel(self.PNL_FILE, sheet_name='Cumulative P&L')
                
                existing_per_slot_data = existing_per_slot_df.to_dict('records')
                existing_day_totals = existing_day_total_df.to_dict('records')
                existing_cumulative = existing_cumulative_df.to_dict('records')
                
                # Get dates already processed
                processed_dates = set(existing_day_total_df['Date'].astype(str).tolist())
            except:
                processed_dates = set()
        else:
            processed_dates = set()
        
        if self.optimal_top_k is None:
            try:
                from scr1_10 import UltimateAllPredictor
                predictor = UltimateAllPredictor()
                self.optimal_top_k = predictor.optimal_top_k
            except:
                self.optimal_top_k = {'FRBD': 25, 'GZBD': 25, 'GALI': 25, 'DSWR': 25}
        
        new_per_slot_rows = []
        new_day_totals = []
        latest_day_breakdown = []
        
        # Process only NEW prediction files (not already in P&L)
        for pred_file in prediction_files:
            date_obj = self.extract_date_from_filename(pred_file)
            if not date_obj: continue
            date_str = date_obj.strftime('%Y-%m-%d')
            
            # Skip if already processed
            if date_str in processed_dates:
                print(f"🔒 SKIP: Date {date_str} already in P&L (immutable)")
                continue
                
            if date_str not in actual_results['date'].dt.date.astype(str).values: 
                print(f"ℹ️ No actual result for {date_str}")
                continue

            # 🔒 P&L MAPPING LOCK
            print(f"🔒 P&L CHECK: Processing {date_str}")
            print(f"   - Prediction file: {pred_file.name}")
            
            # Find actual result for this date
            actual_row = actual_results[actual_results['date'].dt.date.astype(str) == date_str]
            if not actual_row.empty:
                actual_values = {}
                for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                    if slot in actual_row.columns:
                        val = actual_row[slot].iloc[0]
                        actual_values[slot] = val if not pd.isna(val) else 'XX'
                print(f"   - Actual results: {actual_values}")
            else:
                print(f"   - ❌ NO ACTUAL RESULTS FOUND")

            predictions_df = self.load_predictions_file(pred_file)
            if predictions_df is None: continue
            
            actual_for_date = actual_results[actual_results['date'].dt.date.astype(str) == date_str]
            if actual_for_date.empty: continue
            
            per_slot_rows = []
            day_stake = 0
            day_return = 0
            
            for slot_name in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                if slot_name not in actual_for_date.columns: continue
                actual_number = actual_for_date[slot_name].iloc[0]
                if pd.isna(actual_number) or str(actual_number).upper() == 'XX': continue
                
                # Use current optimal K for NEW calculations
                optimal_k = self.optimal_top_k.get(slot_name, 25)
                
                pnl_row = self.calculate_pnl_for_slot(date_str, slot_name, actual_number, 
                                                     predictions_df, optimal_k)
                if pnl_row:
                    per_slot_rows.append(pnl_row)
                    day_stake += pnl_row['Stake (₹)']
                    day_return += pnl_row['Return (₹)']
            
            if not per_slot_rows: continue
            
            new_per_slot_rows.extend(per_slot_rows)
            day_profit = day_return - day_stake
            new_day_totals.append({
                'Date': date_str, 'Total Stake (₹)': day_stake,
                'Total Return (₹)': day_return, 'Total Profit (₹)': day_profit
            })
            
            latest_day_breakdown = per_slot_rows
        
        # Combine existing and new data
        all_per_slot_rows = existing_per_slot_data + new_per_slot_rows
        all_day_totals = existing_day_totals + new_day_totals
        
        # Recalculate cumulative with combined data
        cumulative_data = []
        cumulative_stake = 0
        cumulative_return = 0
        
        for day in sorted(all_day_totals, key=lambda x: x['Date']):
            cumulative_stake += day['Total Stake (₹)']
            cumulative_return += day['Total Return (₹)']
            cumulative_profit = cumulative_return - cumulative_stake
            cumulative_data.append({
                'Date': day['Date'], 'Cumulative Stake (₹)': cumulative_stake,
                'Cumulative Return (₹)': cumulative_return, 'Cumulative Profit (₹)': cumulative_profit
            })
        
        # Save history after processing
        self.save_pnl_history()
        
        if not all_per_slot_rows:
            print("ℹ️ No P&L data processed. Waiting for actual results.")
        
        return all_per_slot_rows, all_day_totals, cumulative_data, latest_day_breakdown
    
    def save_pnl_to_excel(self, per_slot_data, day_total_data, cumulative_data):
        try:
            per_slot_df = pd.DataFrame(per_slot_data)
            day_total_df = pd.DataFrame(day_total_data)
            cumulative_df = pd.DataFrame(cumulative_data)
            with pd.ExcelWriter(self.PNL_FILE, engine='openpyxl') as writer:
                per_slot_df.to_excel(writer, sheet_name='Per-Slot P&L', index=False)
                day_total_df.to_excel(writer, sheet_name='Day Total P&L', index=False)
                cumulative_df.to_excel(writer, sheet_name='Cumulative P&L', index=False)
            return True
        except Exception as e:
            print(f"Error saving P&L file: {e}")
            return False
    
    def rebuild_ledger(self):
        """Explicit rebuild mode - recalculate ALL history"""
        print("⚠️ REBUILDING ENTIRE LEDGER (This will recalculate all past P&L)")
        # Clear history
        self.pnl_history = defaultdict(dict)
        
        # Delete existing files
        if self.PNL_FILE.exists():
            self.PNL_FILE.unlink()
        if self.PNL_HISTORY_FILE.exists():
            self.PNL_HISTORY_FILE.unlink()
        
        # Recalculate everything
        per_slot_data, day_total_data, cumulative_data, latest_day_breakdown = self.process_all_predictions()
        if not per_slot_data: return [], [], [], []
        self.save_pnl_to_excel(per_slot_data, day_total_data, cumulative_data)
        return per_slot_data, day_total_data, cumulative_data, latest_day_breakdown
    
    def run(self, debug=False, rebuild=False):
        if rebuild:
            return self.rebuild_ledger()
        else:
            per_slot_data, day_total_data, cumulative_data, latest_day_breakdown = self.process_all_predictions()
            if not per_slot_data: 
                print("ℹ️ No P&L data to save")
                return [], [], [], []
            self.save_pnl_to_excel(per_slot_data, day_total_data, cumulative_data)
            print(f"✓ P&L saved: {self.PNL_FILE.name}")
            return per_slot_data, day_total_data, cumulative_data, latest_day_breakdown

def main():
    import sys
    rebuild_mode = '--rebuild' in sys.argv
    calculator = PNLCalculator()
    calculator.run(rebuild=rebuild_mode)

if __name__ == "__main__":
    main()
