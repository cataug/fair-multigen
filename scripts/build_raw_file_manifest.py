from pathlib import Path
import csv
import hashlib

ROOT = Path("/home/tahiti/MultiFairGDT")
RAW = ROOT / "data/raw"
OUT = ROOT / "manifests/raw_files.csv"

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

rows = []

for p in sorted(RAW.rglob("*")):
    if not p.is_file():
        continue

    if p.suffix == ".aria2":
        status = "aria2_sidecar"
    elif Path(str(p) + ".aria2").exists():
        status = "incomplete"
    else:
        status = "present"

    rows.append({
        "path": str(p.relative_to(ROOT)),
        "bytes": p.stat().st_size,
        "size_mib": round(p.stat().st_size / 1024**2, 3),
        "status": status,
        "sha256": sha256(p) if status == "present" else "",
    })

with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(
        f,
        fieldnames=["path", "bytes", "size_mib", "status", "sha256"]
    )
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {len(rows)} files to {OUT}")
