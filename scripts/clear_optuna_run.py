#!/usr/bin/env python3
"""
Clear a given run/study from Optuna's persistent SQLite storage.

Usage:
    python clear_optuna_run.py <study_name> [--db <database_url>]

Example:
    python clear_optuna_run.py my_study_name
    python clear_optuna_run.py my_study_name --db "sqlite:///path/to/database.db"
"""

import argparse
import sys
from pathlib import Path

import optuna
from optuna.storages import RDBStorage


def clear_optuna_run(study_name: str, storage_url: str) -> bool:
    """
    Clear a study from Optuna's persistent storage.

    Args:
        study_name: Name of the study to delete
        storage_url: URL of the SQLite storage (e.g., "sqlite:///path/to/db.db")

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create storage connection
        storage = RDBStorage(storage_url)

        # Try to load the study to check if it exists
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            n_trials = len(study.trials)
            print(f"Found study '{study_name}'")
            print(f"This study has {n_trials} trials that will be deleted.")
        except optuna.exceptions.StudyDoesNotExist:
            print(f"❌ Study '{study_name}' not found in database.")
            print("\nListing all available studies...")
            try:
                # Try to list existing studies
                studies = optuna.get_all_study_summaries(storage=storage)
                if studies:
                    print("\nAvailable studies:")
                    for s in studies:
                        print(f"  - {s.study_name} ({len(s.trials)} trials)")
                else:
                    print("  (No studies found in database)")
            except Exception:
                print("  (Could not list studies)")
            return False

        # Confirm deletion
        response = input("\nAre you sure you want to delete this study? (yes/no): ")
        if response.lower() not in ('yes', 'y'):
            print("Cancelled.")
            return False

        # Delete the study using optuna's delete function
        optuna.delete_study(study_name=study_name, storage=storage)
        print(f"✅ Successfully deleted study '{study_name}'")
        return True

    except FileNotFoundError as e:
        print(f"❌ Database file not found: {e}")
        return False
    except Exception as e:
        print(f"❌ Error deleting study: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Clear a study from Optuna's persistent SQLite storage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clear_optuna_run.py my_study_name
  python clear_optuna_run.py my_study_name --db "sqlite:///runs/optuna_persist/pi_optimization.db"
        """
    )

    parser.add_argument(
        "study_name",
        type=str,
        help="Name of the study to delete"
    )

    parser.add_argument(
        "--db",
        type=str,
        default="sqlite:///runs/optuna_persist/pi_optimization.db",
        help="SQLite storage URL (default: %(default)s)"
    )

    args = parser.parse_args()

    print(f"Database: {args.db}")
    print(f"Study: {args.study_name}")
    print("-" * 70)

    success = clear_optuna_run(args.study_name, args.db)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
