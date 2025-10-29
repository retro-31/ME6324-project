#!/usr/bin/env python3
"""
Scrape images for CORROSION and NOCORROSION classes and save to data/<CLASS>/.

Uses DuckDuckGo image search (no API key required) via `duckduckgo_search` to fetch image URLs,
then downloads them with `requests`, validates with Pillow, and stores with unique names.

Example:
  python scrape_images.py --per-query 50 --max-per-class 300
  python scrape_images.py --dry-run  # lists what would be fetched without downloading

Note: Always respect website terms of service and robots.txt. This script is intended for
educational and research use. You are responsible for ensuring that downloaded images are
used in compliance with copyrights and applicable policies.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Dict, Set, Tuple

# Third-party deps (documented in requirements_scrape.txt)
try:
    from duckduckgo_search import DDGS  # type: ignore
except Exception as e:  # pragma: no cover
    DDGS = None  # allow script to show a clear error if missing

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # Provide a no-op fallback if tqdm isn't installed
    def tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable if iterable is not None else []


DEFAULT_CORROSION_QUERIES = [
    "corroded metal",
    "rust corrosion",
    "corroded pipe",
    "corroded steel beam",
    "rusted machinery",
    "corroded surface close up",
]

DEFAULT_NOCORROSION_QUERIES = [
    "clean metal surface",
    "polished stainless steel",
    "new metal pipe",
    "shiny metal machinery",
    "smooth aluminum surface",
    "freshly painted metal",
]

ALLOWED_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


@dataclass
class DownloadResult:
    saved: int = 0
    skipped: int = 0
    failed: int = 0


def build_http() -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/118.0 Safari/537.36"
            )
        }
    )
    return sess


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def unique_name(url: str, idx: int, content_type: str | None) -> str:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    ext = ALLOWED_CONTENT_TYPES.get(content_type or "", ".jpg")
    return f"img_{idx:05d}_{h}{ext}"


def is_valid_image(data: bytes, min_size: int) -> Tuple[bool, Tuple[int, int] | None]:
    with contextlib.ExitStack() as stack:
        bio = stack.enter_context(io.BytesIO(data))
        try:
            im = stack.enter_context(Image.open(bio))
            im.verify()  # type: ignore[attr-defined]
        except Exception:
            return False, None
    # Reopen to get size (verify() leaves file in an unusable state for getbbox)
    try:
        with Image.open(io.BytesIO(data)) as im2:
            w, h = im2.size
            if min(w, h) < min_size:
                return False, (w, h)
            return True, (w, h)
    except Exception:
        return False, None


def fetch_image_urls(query: str, per_query: int, safesearch: str = "Moderate") -> List[str]:
    if DDGS is None:
        raise RuntimeError(
            "duckduckgo_search is not installed. Please run: pip install -r requirements_scrape.txt"
        )
    urls: List[str] = []
    # region="wt-wt" means global; safesearch options: Off, Moderate, Strict
    with DDGS() as ddgs:
        for item in ddgs.images(
            keywords=query,
            region="wt-wt",
            safesearch=safesearch,
            max_results=per_query,
        ):
            url = item.get("image") or item.get("thumbnail")
            if isinstance(url, str):
                urls.append(url)
    return urls


def download_and_save(
    session: requests.Session,
    url: str,
    out_dir: str,
    idx: int,
    timeout: float,
    min_size: int,
) -> bool:
    try:
        r = session.get(url, timeout=timeout, stream=False)
    except Exception:
        return False
    if r.status_code != 200:
        return False
    content_type = r.headers.get("Content-Type", "").split(";")[0].strip()
    if content_type not in ALLOWED_CONTENT_TYPES:
        # Attempt to read and validate anyway; if valid, default to jpg
        try:
            data = r.content
        except Exception:
            return False
        ok, _ = is_valid_image(data, min_size)
        if not ok:
            return False
        filename = unique_name(url, idx, "image/jpeg")
        path = os.path.join(out_dir, filename)
        try:
            with open(path, "wb") as f:
                f.write(data)
            return True
        except Exception:
            with contextlib.suppress(Exception):
                if os.path.exists(path):
                    os.remove(path)
            return False
    else:
        try:
            data = r.content
        except Exception:
            return False
        ok, _ = is_valid_image(data, min_size)
        if not ok:
            return False
        filename = unique_name(url, idx, content_type)
        path = os.path.join(out_dir, filename)
        try:
            with open(path, "wb") as f:
                f.write(data)
            return True
        except Exception:
            with contextlib.suppress(Exception):
                if os.path.exists(path):
                    os.remove(path)
            return False


def collect_for_class(
    cls_name: str,
    queries: Iterable[str],
    out_root: str,
    per_query: int,
    max_per_class: int | None,
    timeout: float,
    min_size: int,
    dry_run: bool,
) -> DownloadResult:
    out_dir = os.path.join(out_root, cls_name)
    ensure_dir(out_dir)
    session = build_http()

    result = DownloadResult()
    seen: Set[str] = set()
    saved_total = 0

    for q in queries:
        if max_per_class is not None and saved_total >= max_per_class:
            break
        if dry_run:
            # In dry-run, don't hit the network at all. Assume up to per_query items considered.
            to_count = per_query if max_per_class is None else min(per_query, max(0, max_per_class - saved_total))
            result.skipped += to_count
            saved_total += to_count
            continue

        try:
            urls = fetch_image_urls(q, per_query)
        except Exception as e:
            print(f"[WARN] Search failed for '{q}': {e}")
            result.failed += 1
            continue
        # Dedup by URL
        new_urls = [u for u in urls if u not in seen]
        for i, url in enumerate(tqdm(new_urls, desc=f"{cls_name}: {q}", leave=False)):
            if max_per_class is not None and saved_total >= max_per_class:
                break
            seen.add(url)
            ok = download_and_save(session, url, out_dir, idx=saved_total, timeout=timeout, min_size=min_size)
            if ok:
                result.saved += 1
                saved_total += 1
            else:
                result.failed += 1
        # polite delay between queries
        time.sleep(0.5)
    return result


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape images into data/CORROSION and data/NOCORROSION")
    p.add_argument("--output-root", default="data", help="Root folder to save images (default: data)")
    p.add_argument("--per-query", type=int, default=100, help="Max images to fetch per search query")
    p.add_argument("--max-per-class", type=int, default=300, help="Max images to save per class")
    p.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds per request")
    p.add_argument("--min-size", type=int, default=224, help="Minimum image dimension (shorter side) in pixels")
    p.add_argument("--dry-run", action="store_true", help="Do not download; just list queries and counts")
    p.add_argument("--corrosion", nargs="*", default=DEFAULT_CORROSION_QUERIES, help="Override corrosion queries")
    p.add_argument(
        "--nocorrosion", nargs="*", default=DEFAULT_NOCORROSION_QUERIES, help="Override non-corrosion queries"
    )
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    print("Configured queries:")
    print(f"  CORROSION:   {args.corrosion}")
    print(f"  NOCORROSION: {args.nocorrosion}")
    print(
        f"Settings: per_query={args.per_query}, max_per_class={args.max_per_class}, min_size={args.min_size}, dry_run={args.dry_run}"
    )

    total = {"CORROSION": DownloadResult(), "NOCORROSION": DownloadResult()}

    total["CORROSION"] = collect_for_class(
        "CORROSION",
        args.corrosion,
        args.output_root,
        per_query=args.per_query,
        max_per_class=args.max_per_class,
        timeout=args.timeout,
        min_size=args.min_size,
        dry_run=args.dry_run,
    )
    total["NOCORROSION"] = collect_for_class(
        "NOCORROSION",
        args.nocorrosion,
        args.output_root,
        per_query=args.per_query,
        max_per_class=args.max_per_class,
        timeout=args.timeout,
        min_size=args.min_size,
        dry_run=args.dry_run,
    )

    print("\nSummary:")
    for k, v in total.items():
        print(f"  {k}: saved={v.saved}, skipped={v.skipped}, failed={v.failed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
