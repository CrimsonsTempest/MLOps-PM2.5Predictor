import sqlite3
import argparse
import json
import sys
from pathlib import Path

# Dynamically resolve the project root based on this script's location
# __file__ = src/mlflow-metadata/extract_mlflow_run.py
# .parent = src/mlflow-metadata/
# .parent.parent = src/
# .parent.parent.parent = MLOps-PM2.5Predictor/ (Project Root)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"

def extract_run_metadata(db_path: str, run_id: str) -> dict:
    """Extracts run info, parameters, metrics, and tags from MLflow SQLite DB."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 1. Extract base run info
        cursor.execute("SELECT * FROM runs WHERE run_uuid = ?", (run_id,))
        run_row = cursor.fetchone()
        
        if not run_row:
            print(f"Error: Run ID '{run_id}' not found in database '{db_path}'.", file=sys.stderr)
            sys.exit(1)
            
        metadata = {
            "run_info": dict(run_row),
            "params": {},
            "metrics": {},
            "tags": {}
        }
        
        # 2. Extract Parameters
        cursor.execute("SELECT key, value FROM params WHERE run_uuid = ?", (run_id,))
        for row in cursor.fetchall():
            metadata["params"][row["key"]] = row["value"]
            
        # 3. Extract Metrics
        cursor.execute("SELECT key, value FROM metrics WHERE run_uuid = ? ORDER BY timestamp ASC", (run_id,))
        for row in cursor.fetchall():
            metadata["metrics"][row["key"]] = row["value"]
            
        # 4. Extract Tags
        cursor.execute("SELECT key, value FROM tags WHERE run_uuid = ?", (run_id,))
        for row in cursor.fetchall():
            metadata["tags"][row["key"]] = row["value"]
            
        return metadata

    except sqlite3.Error as e:
        print(f"SQLite error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MLflow run metadata directly from SQLite.")
    parser.add_argument("run_id", help="The MLflow Run ID (run_uuid)")
    parser.add_argument("--db", default="mlflow.db", help="Path to the MLflow SQLite database file (default: mlflow.db)")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_CONFIG_DIR), 
                        help="Directory to save the JSON file (defaults to MLOps-PM2.5Predictor/config)")
    
    args = parser.parse_args()
    
    # Extract the data
    run_data = extract_run_metadata(args.db, args.run_id)
    
    # Prepare output directory and filename
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = out_dir / f"mlflow_run_{args.run_id}_meta.json"
        
    # Write to output file
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=2)
        print(f"Success: Run metadata dumped to\n -> {out_file}")
    except IOError as e:
        print(f"Error writing to file '{out_file}': {e}", file=sys.stderr)
        sys.exit(1)