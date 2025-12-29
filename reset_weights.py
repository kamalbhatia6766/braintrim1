import json
from pathlib import Path

def reset_weights():
    PROJECT_DIR = Path(__file__).resolve().parent
    WEIGHTS_FILE = PROJECT_DIR / "model_weights.json"
    
    model_weights = {}
    scripts = ['scr1', 'scr2', 'scr3', 'scr4', 'scr6', 'scr7', 'scr8', 'scr9', 'scr10', 'scr11', 'scr12', 'scr13', 'scr14']
    
    for script in scripts:
        for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
            model_id = f"{script}_{slot}"
            model_weights[model_id] = {
                'weight': 0.3, 
                'overall_accuracy': 0.0, 
                'total_hits': 0, 
                'total_attempts': 0,
                'weight_history': [0.3], 
                'performance_history': [],
                'recent_hits': 0, 
                'recent_attempts': 0, 
                'consistency_score': 0.5,
                'clamp_status': '',
                'pre_clamp_weight': 0.3
            }
    
    data = {'weights': model_weights, 'processed_dates': []}
    
    with open(WEIGHTS_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("✓ Weights reset to default (including new scr11, scr12, scr13, scr14)")

if __name__ == "__main__":
    reset_weights()