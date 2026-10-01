#!/usr/bin/env python3
"""素材抓取（公有领域/CC0）— 供 AI 修图产线 v0 使用，记录来源与许可。

来源：
  · Met Museum Open Access（CC0）：高清绘画（对拍集 GT）+ 器皿（商品图素材）
  · Wikimedia Commons（Public domain）：历史老照片

产物：
  materials/manifest.json  来源 / 许可 / 尺寸 / sha256
  materials/pairs/gt/*.png, materials/products/*.jpg, materials/historical/*.jpg

CLI:
    python3 inference/photo/tools/fetch_assets.py --outdir /mnt/data/ai_workspace/outputs/photo-retouch/materials
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = "cland-asset-fetch/1.0 (C-Land photo-retouch v0; contact: sage)"
MET_OBJECT = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}"
MET_SEARCH = "https://collectionapi.metmuseum.org/public/collection/v1/search"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# name, objectID, 用途, 退化配方（对拍集）
MET_PAIRS = [
    ("pair01_the_harvesters", 435809, "Pieter Bruegel the Elder - The Harvesters (1565)",
     "downscale:2.2,blur:1.1,noise:9,scratch:5,vignette:0.18,fade:0.12,jpeg:84"),
    ("pair02_wheat_field", 436535, "Van Gogh - Wheat Field with Cypresses (1889)",
     "downscale:2.6,gray:1,sepia:70,blur:1.4,noise:11,scratch:6,frame:0.02,vignette:0.22,jpeg:80"),
    ("pair03_great_wave", 45434, "Hokusai - Under the Wave off Kanagawa",
     "downscale:3.2,blur:1.0,noise:8,scratch:4,vignette:0.15,fade:0.10,jpeg:78"),
]

MET_PRODUCTS = [
    ("prod01_potpourri_jar", 200876, "Potpourri jar (Met Open Access)"),
    ("prod02_glass_cooler", 200840, "Glass cooler (seau à verre)"),
    ("prod03_dish_death_of_saul", 198704, "Dish depicting The Death of Saul"),
]

COMMONS_HISTORICAL = [
    ("hist01_migrant_mother", "File:Lange-MigrantMother02.jpg",
     "Dorothea Lange - Migrant Mother (1936), FSA / Library of Congress"),
    ("hist02_lincoln", "File:Abraham Lincoln O-77 matte collodion print.jpg",
     "Alexander Gardner - Abraham Lincoln (1863)"),
]


def _get(url: str, timeout: int = 120, attempts: int = 4) -> bytes:
    """用 curl 下载（比 urllib 在弱网/大文件下稳定，断线自动重试）。"""
    import subprocess
    last: Exception | None = None
    for i in range(attempts):
        cmd = ["curl", "-sSL", "--fail", "--connect-timeout", "20", "-m", str(timeout),
               "-A", UA, url]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0 and r.stdout:
            return r.stdout
        last = RuntimeError(f"curl rc={r.returncode}: {r.stderr.decode(errors='ignore')[:200]}")
        print(f"[retry {i + 1}/{attempts}] {last}", flush=True)
        time.sleep(2 * (i + 1))
    raise last  # type: ignore[misc]


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _save(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _fetch(url: str, path: Path, timeout: int = 120) -> bytes:
    """幂等下载：本地已有非空文件则直接复用。"""
    if path.exists() and path.stat().st_size > 0:
        print(f"[skip] exists: {path.name} ({path.stat().st_size // 1024}KB)")
        return path.read_bytes()
    data = _get(url, timeout=timeout)
    _save(path, data)
    return data


def _met_object(oid: int) -> dict:
    return json.loads(_get(MET_OBJECT.format(oid=oid)))


def fetch_met_pairs(outdir: Path, records: list) -> None:
    for name, oid, title, degrade in MET_PAIRS:
        o = _met_object(oid)
        if not o.get("isPublicDomain"):
            raise RuntimeError(f"{oid} 非公有领域，跳过")
        url = o["primaryImage"]
        gt_path = outdir / "pairs" / "gt" / f"{name}.jpg"
        data = _fetch(url, gt_path)
        records.append({
            "name": name, "group": "pairs", "met_object_id": oid, "title": title,
            "license": "CC0 (Met Open Access)", "source_url": url, "page": o.get("objectURL"),
            "local": str(gt_path), "bytes": len(data), "sha256": _sha256(data),
            "degrade": degrade,
            "download_ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        print(f"[pairs] {name}: {len(data) // 1024}KB <- {url}")


def fetch_met_products(outdir: Path, records: list) -> None:
    for name, oid, title in MET_PRODUCTS:
        o = _met_object(oid)
        if not o.get("isPublicDomain"):
            raise RuntimeError(f"{oid} 非公有领域，跳过")
        url = o["primaryImageSmall"]
        p = outdir / "products" / f"{name}.jpg"
        data = _fetch(url, p)
        records.append({
            "name": name, "group": "products", "met_object_id": oid, "title": title,
            "license": "CC0 (Met Open Access)", "source_url": url, "page": o.get("objectURL"),
            "local": str(p), "bytes": len(data), "sha256": _sha256(data),
            "download_ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        print(f"[prod] {name}: {len(data) // 1024}KB <- {url}")


def fetch_commons_historical(outdir: Path, records: list) -> None:
    for name, title, desc in COMMONS_HISTORICAL:
        params = {
            "action": "query", "format": "json", "titles": title, "prop": "imageinfo",
            "iiprop": "url|size|extmetadata", "iiurlwidth": "1920",
        }
        data = json.loads(_get(COMMONS_API + "?" + urllib.parse.urlencode(params)))
        pages = data["query"]["pages"]
        page = next(iter(pages.values()))
        ii = page["imageinfo"][0]
        url = ii.get("thumburl") or ii["url"]
        p = outdir / "historical" / f"{name}.jpg"
        img = _fetch(url, p, timeout=300)
        em = ii.get("extmetadata", {})
        records.append({
            "name": name, "group": "historical", "commons_title": title, "title": desc,
            "license": em.get("LicenseShortName", {}).get("value", "Public domain"),
            "artist": (em.get("Artist", {}).get("value") or "")[:120],
            "source_url": url, "page": f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
            "local": str(p), "bytes": len(img), "sha256": _sha256(img),
            "download_ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        print(f"[hist] {name}: {len(img) // 1024}KB <- {url}")


def main():
    ap = argparse.ArgumentParser(description="抓取 CC0/公有领域素材")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    records: list = []
    fetch_met_pairs(outdir, records)
    fetch_met_products(outdir, records)
    fetch_commons_historical(outdir, records)
    manifest = {
        "generated_ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "usage": "AI 修图产线 v0 素材（仅公有领域/CC0；对拍集由 degrade.py 退化生成）",
        "records": records,
    }
    mp = outdir / "manifest.json"
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[manifest] {len(records)} assets -> {mp}")


if __name__ == "__main__":
    main()
