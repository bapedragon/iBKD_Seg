"""Pinned, unmodified upstream sources and verified public initial weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

SOURCES = {
    "segmenter": ("https://github.com/rstrudel/segmenter.git", "20d1bfad354165ee45c3f65972a4d9c131f58d53"),
    "mmcv": ("https://github.com/open-mmlab/mmcv.git", "db097bd1e97fc446a7551c715970611d2fcc848d"),
    "mmseg": ("https://github.com/open-mmlab/mmsegmentation.git", "26032167e005f391aad1bf151a7e8f9a98973266"),
}
WEIGHTS = {
    "vit_large_384.npz": {
        "url": "https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz",
        "bytes": 1218991142,
        "sha256": "8700cf38ba82082d95347ad16ea3294d588b359077a7c46da9da6618165c3288",
    },
    "deeplabv3_r101.pth": {
        "url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x1024_80k_cityscapes/deeplabv3_r101-d8_512x1024_80k_cityscapes_20200606_113503-9e428899.pth",
        "bytes": 348988299,
        "sha256": "9e428899b279f29964cec79ab21bb19193328b8c4d42c0db49ff9070e9ab3b2d",
    },
}
TINY_WEIGHT = {
    "url": "https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz",
    "bytes": 23226422,
    "sha256": "4b99893dc1a5a2a7d9ad119671c20559850f865e1fa17ed23401a3fefa7fedc9",
}
SMALL_WEIGHT = {
    "url": "https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz",
    "bytes": 88851254,
    "sha256": "9e4155229ce0b767ef43ff1e3a9f4308a69b61f44969e0605df5f39047f2e7a8",
}


def weight_manifest(student="large", *, include_teacher=True):
    if student == "large":
        result = dict(WEIGHTS)
    elif student == "tiny":
        result = {"vit_tiny_384.npz": TINY_WEIGHT,
                  "deeplabv3_r101.pth": WEIGHTS["deeplabv3_r101.pth"]}
    elif student == "small":
        result = {"vit_small_384.npz": SMALL_WEIGHT,
                  "deeplabv3_r101.pth": WEIGHTS["deeplabv3_r101.pth"]}
    else:
        raise ValueError(f"Unsupported student asset set: {student}")
    if not include_teacher:
        result.pop("deeplabv3_r101.pth")
    return result


DEPENDENCIES = ["timm==0.4.12", "addict==2.4.0", "yapf==0.40.1",
                "importlib-metadata==8.7.0", "platformdirs==4.3.8", "tomli==2.2.1",
                "zipp==3.23.0", "einops==0.8.1", "opencv-python-headless==4.13.0.92",
                "prettytable==3.16.0", "wcwidth==0.2.13", "packaging==25.0",
                "PyYAML==6.0.2", "matplotlib==3.10.6"]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root, *, student="large", include_teacher=True):
    report = {"sources": {}, "weights": {}}
    for name, (url, commit) in SOURCES.items():
        path = root / name
        actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if actual != commit:
            raise RuntimeError(f"Wrong upstream commit: {name}: {actual}")
        subprocess.run(["git", "-C", str(path), "diff", "--exit-code", "HEAD", "--"], check=True)
        files = subprocess.check_output(["git", "-C", str(path), "ls-files"], text=True).splitlines()
        # Also detect untracked source files which could shadow upstream imports.
        extra = subprocess.check_output(["git", "-C", str(path), "ls-files", "--others", "--exclude-standard"], text=True).splitlines()
        if any(p.endswith((".py", ".yml", ".yaml", ".json")) for p in extra):
            raise RuntimeError(f"Untracked upstream source files: {name}")
        hashes = {p: sha(path / p) for p in files if p.endswith((".py", ".yml", ".yaml"))}
        report["sources"][name] = {"url": url, "commit": actual, "source_hashes": hashes}
    for name, expected in weight_manifest(student, include_teacher=include_teacher).items():
        path = root / "weights" / name
        if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise RuntimeError(f"Public checkpoint hash mismatch: {path}")
        report["weights"][name] = expected
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--student", choices=("large", "tiny", "small"), default="large")
    parser.add_argument("--student-only", action="store_true", help="Prepare encoder assets for evaluation without a teacher")
    args = parser.parse_args()
    root = args.cache_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name, (url, commit) in SOURCES.items():
        path = root / name
        if not path.exists():
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(path)], check=True)
            subprocess.run(["git", "-C", str(path), "checkout", "--detach", commit], check=True)
    # Legacy timm remains isolated from the project's timm 1.x environment.
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                    "--target", str(root / "deps"), *DEPENDENCIES[1:]], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                    "--target", str(root / "deps"), "--no-deps", DEPENDENCIES[0]], check=True)
    (root / "weights").mkdir(exist_ok=True)
    for name, info in weight_manifest(args.student, include_teacher=not args.student_only).items():
        path = root / "weights" / name
        if not path.exists():
            temp = path.with_suffix(path.suffix + ".part")
            print(f"[OFFICIAL_DOWNLOAD] {name} bytes={info['bytes']}", flush=True)
            with urllib.request.urlopen(info["url"], timeout=120) as response, temp.open("wb") as out:
                for chunk in iter(lambda: response.read(8 << 20), b""):
                    out.write(chunk)
            if temp.stat().st_size != info["bytes"] or sha(temp) != info["sha256"]:
                raise RuntimeError(f"Downloaded checkpoint checksum mismatch: {name}")
            temp.replace(path)
    report = verify(root, student=args.student, include_teacher=not args.student_only)
    filename = "provenance.json" if args.student == "large" else f"provenance_{args.student}.json"
    if args.student_only:
        filename = f"provenance_{args.student}_student_only.json"
    (root / filename).write_text(json.dumps(report, indent=2) + "\n")
    print(f"[OFFICIAL_ASSETS_READY] sources=3 weights={len(report['weights'])} hashes=passed", flush=True)


if __name__ == "__main__":
    main()
