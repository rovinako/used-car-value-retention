"""Run the saved-data analysis. Scraping is a separate, optional step."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
for script in ["prepare_data.py", "train_models.py", "make_report.py"]:
    subprocess.run([sys.executable, str(root / "src" / script)], cwd=root, check=True)
