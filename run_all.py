"""Run the complete pipeline: python run_all.py"""
import runpy
import build_dataset

if __name__ == "__main__":
    build_dataset.main()
    runpy.run_path("train_evaluate.py")