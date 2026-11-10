"""Join the two Drive downloads and verify the frozen restore archive."""

import argparse
import hashlib
import json
from pathlib import Path


MANIFEST = Path(__file__).resolve().parents[1] / "results" / "restore_kit_manifest.json"


def assemble(downloads: Path) -> Path:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    parts = [(downloads / entry["name"], entry) for entry in manifest["parts"]]
    for path, entry in parts:
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Wrong size for {path.name}. Download it again from Drive.")

    output = downloads / manifest["file"]
    archive_hash = hashlib.sha256()
    with output.open("xb") as destination:
        try:
            for path, entry in parts:
                part_hash = hashlib.sha256()
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        part_hash.update(block)
                        archive_hash.update(block)
                        destination.write(block)
                if part_hash.hexdigest() != entry["sha256"]:
                    raise ValueError(f"SHA-256 mismatch for {path.name}")
            if archive_hash.hexdigest() != manifest["sha256"]:
                raise ValueError("SHA-256 mismatch for the joined restore ZIP")
        except BaseException:
            destination.close()
            output.unlink()
            raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("downloads", type=Path, help="Directory holding both Drive parts")
    args = parser.parse_args()
    try:
        output = assemble(args.downloads.expanduser())
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Verified archive: {output}")


if __name__ == "__main__":
    main()
