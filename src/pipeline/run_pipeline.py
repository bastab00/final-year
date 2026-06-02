import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

PIPELINE = [

    # =========================
    # DATA COLLECTION
    # =========================
    ROOT_DIR / "src" / "data_collection" / "collect_1000_openalex.py",

    # =========================
    # PREPROCESSING
    # =========================
    ROOT_DIR / "src" / "preprocessing" / "preprocess_dataset.py",

    ROOT_DIR / "src" / "preprocessing" / "merge_latest.py",

    # =========================
    # MODEL TRAINING
    # =========================
    ROOT_DIR / "src" / "models" / "train_baseline_classifier.py",

    ROOT_DIR / "src" / "models" / "train_gaussian_svm.py",

    ROOT_DIR / "src" / "models" / "train_random_forest.py",

    ROOT_DIR / "src" / "models" / "train_xgboost_classifier.py",

    ROOT_DIR / "src" / "models" / "train_sbert_classifier.py",

    ROOT_DIR / "src" / "models" / "train_sbert_lr_classifier.py",

    # =========================
    # EVALUATION
    # =========================
    ROOT_DIR / "src" / "models" / "generate_results_table.py"
]


def run_script(script_path):

    print("\n" + "=" * 80)
    print(f"RUNNING: {script_path.name}")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(script_path)]
    )

    if result.returncode != 0:

        print(
            f"\n[FAILED] {script_path.name}"
        )

        raise RuntimeError(
            f"Pipeline stopped at: {script_path.name}"
        )

    print(
        f"\n[SUCCESS] {script_path.name}"
    )


def main():

    start_time = time.time()

    print("\n" + "=" * 80)
    print("STARTING AUTOMATED RETRAINING PIPELINE")
    print("=" * 80)

    for script in PIPELINE:
        run_script(script)

    elapsed = (
        time.time() - start_time
    ) / 60

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Total Time: {elapsed:.2f} minutes")
    print("=" * 80)

    print("\nGenerated Artifacts:")
    print("------------------------------------")
    print("dataset_latest_version.csv")
    print("baseline_lr_model_latest.pkl")
    print("gaussian_svm_model_latest.pkl")
    print("rf_model_latest.pkl")
    print("xgb_model_latest.pkl")
    print("sbert_svm_model_latest.pkl")
    print("sbert_lr_classifier_latest.pkl")
    print("model_comparison_results_latest.csv")
    print("------------------------------------")


if __name__ == "__main__":
    main()