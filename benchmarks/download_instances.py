"""
benchmarks/download_instances.py
---------------------------------
Download and cache standard Netlib LP and MIPLIB instances for benchmarking.

Instances:
  Netlib LP:
    - afiro.mps      (32 vars, 27 cons)
    - brandy.mps     (249 vars, 220 cons)
    - adlittle.mps   (97 vars, 56 cons)
    - blending.mps   (simple blending LP)
  MIPLIB:
    - exmip1.mps     (8 vars, 6 cons, 2 binary)
    - p0033.mps      (33 binary vars, knapsack/packing)
    - p0548.mps      (548 binary vars, knapsack/packing)
"""

from __future__ import annotations

import os
import shutil
import urllib.request

INSTANCES_DIR = os.path.join(os.path.dirname(__file__), "instances")
TEST_INSTANCES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "instances")

URLS = {
    # Netlib LPs
    "afiro.mps": "https://raw.githubusercontent.com/coin-or-tools/Data-Sample/master/afiro.mps",
    "brandy.mps": "https://raw.githubusercontent.com/coin-or-tools/Data-Sample/master/brandy.mps",
    "adlittle.mps": "https://raw.githubusercontent.com/ERGO-Code/HiGHS/master/check/instances/adlittle.mps",
    "blending.mps": "https://raw.githubusercontent.com/ERGO-Code/HiGHS/master/check/instances/blending.mps",
    # MIPs
    "exmip1.mps": "https://raw.githubusercontent.com/coin-or-tools/Data-Sample/master/exmip1.mps",
    "p0033.mps": "https://raw.githubusercontent.com/coin-or-tools/Data-Sample/master/p0033.mps",
    "p0548.mps": "https://raw.githubusercontent.com/coin-or-tools/Data-Sample/master/p0548.mps",
}


def download_all_instances(target_dir: str = INSTANCES_DIR) -> dict[str, str]:
    """Ensure all benchmark instances are downloaded and present in target_dir."""
    os.makedirs(target_dir, exist_ok=True)
    paths = {}

    for name, url in URLS.items():
        dst = os.path.join(target_dir, name)
        # Check if already in target_dir
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            paths[name] = dst
            continue

        # Check if exists in tests/instances/
        src_test = os.path.join(TEST_INSTANCES_DIR, name)
        if os.path.exists(src_test) and os.path.getsize(src_test) > 0:
            shutil.copy2(src_test, dst)
            print(f"Copied from tests/instances: {name}")
            paths[name] = dst
            continue

        # Download from URL
        print(f"Downloading {name} from {url}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp, open(dst, "wb") as f:
                f.write(resp.read())
            print(f"  Saved {name} ({os.path.getsize(dst)} bytes)")
            paths[name] = dst
        except Exception as e:
            print(f"  Failed to download {name}: {e}")

    return paths


if __name__ == "__main__":
    download_all_instances()
