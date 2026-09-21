import subprocess
import sys
import time


def run_step(step_num, description, script_name):
    print(f"\n==================================================")
    print(f"[{step_num}/6] {description} ({script_name})")
    print(f"==================================================")
    start = time.time()
    result = subprocess.run(
        [sys.executable, script_name], capture_output=False)
    if result.returncode != 0:
        print(
            f"\n[ERROR] Step failed: {script_name} returned exit code {result.returncode}")
        sys.exit(result.returncode)
    duration = time.time() - start
    print(f"[COMPLETED] {script_name} executed in {duration:.1f}s")


def main():
    print("==================================================")
    print("Climate Displacement Foresight - Full Pipeline")
    print("==================================================")

    pipeline = [
        ("1", "World Bank Indicator Fetch", "01_fetch_worldbank_data.py"),
        ("2", "IDMC Displacement Aggregation", "02_fetch_idmc_displacement.py"),
        ("3", "Panel Integration & Feature Engineering",
         "03_merge_and_engineer_features.py"),
        ("4", "Exploratory Data Analysis", "04_exploratory_data_analysis.py"),
        ("5", "Model Benchmarking & Evaluation", "05_train_models.py"),
        ("6", "Policy Foresight Watchlist Generation",
         "06_generate_foresight_report.py")
    ]

    total_start = time.time()
    for step_num, desc, script in pipeline:
        run_step(step_num, desc, script)

    total_time = time.time() - total_start
    print(f"\nPipeline finished successfully in {total_time:.1f}s!")
    print("Launch interactive monitor: streamlit run app.py")


if __name__ == '__main__':
    main()
