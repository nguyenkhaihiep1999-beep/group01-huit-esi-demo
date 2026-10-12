"""Verify every packaged file using only Python's standard library."""
import hashlib
import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "package_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Missing or invalid bundle path: " + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Bundle checksum mismatch: " + name)
    print("PASS:", len(manifest["files"]), "packaged files match SHA-256.")

if __name__ == "__main__":
    main()
