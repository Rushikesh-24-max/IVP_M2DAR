"""Regression checks for the narrowly scoped wheel-metadata repair."""
import base64
import csv
import hashlib
import importlib.util
import io
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "environment/check_decord.py"
spec = importlib.util.spec_from_file_location("decord_metadata_repair", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DecordMetadata(unittest.TestCase):
    def test_only_tag_and_record_hash_change(self):
        wheel = b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp36-cp36m-manylinux2010_x86_64\n"
        record = b"decord/libdecord.so,sha256=do-not-change,100\ndecord-0.6.0.dist-info/WHEEL,sha256=old,10\ndecord-0.6.0.dist-info/RECORD,,\n"
        updated, listing, changed = module.repaired_contents(wheel, record, "decord-0.6.0.dist-info/WHEEL")
        self.assertTrue(changed)
        self.assertEqual(updated, wheel.replace(b"cp36-cp36m", b"py3-none"))
        rows = list(csv.reader(io.StringIO(listing.decode())))
        self.assertEqual(rows[0], ["decord/libdecord.so", "sha256=do-not-change", "100"])
        digest = base64.urlsafe_b64encode(hashlib.sha256(updated).digest()).rstrip(b"=").decode()
        self.assertEqual(rows[1][1:], ["sha256=" + digest, str(len(updated))])
        self.assertEqual(rows[2][1:], ["", ""])
        self.assertFalse(module.repaired_contents(updated, listing, "decord-0.6.0.dist-info/WHEEL")[2])

    def test_unrecognized_platform_or_missing_record_rejected(self):
        with self.assertRaises(ValueError):
            module.repaired_contents(b"Tag: cp310-cp310-win_amd64\n", b"", "WHEEL")
        with self.assertRaises(ValueError):
            module.repaired_contents(b"Tag: cp36-cp36m-manylinux2010_x86_64\n", b"other,,\n", "WHEEL")


if __name__ == "__main__":
    unittest.main()
