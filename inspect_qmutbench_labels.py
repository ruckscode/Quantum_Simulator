from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
csv_path = Path.home() / "Downloads" / "all_data.csv"

print("=== ORIGINAL QMutBench CSV ===")
if csv_path.exists():
    df = pd.read_csv(csv_path, nrows=5)
    print("Columns:", df.columns.tolist())
    print("\nSample rows:")
    print(df.to_string(index=False))
else:
    print(f"CSV not found: {csv_path}")

print("\n=== PROJECT DOCUMENTATION MATCHES ===")
terms = ("WOO", "OPO", "Survivor", "Killed")
extensions = {".md", ".txt", ".rst", ".csv"}

for path in root.rglob("*"):
    if not path.is_file():
        continue
    if any(part in {".venv", ".git", "__pycache__", "node_modules"}
           for part in path.parts):
        continue
    if path.suffix.lower() not in extensions:
        continue
    if path.name.lower() == "all_data.csv":
        continue

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f, 1):
                if any(term.lower() in line.lower() for term in terms):
                    print(f"{path.relative_to(root)}:{line_no}: {line.strip()[:250]}")
    except OSError:
        pass
