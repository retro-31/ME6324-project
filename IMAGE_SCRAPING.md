# Image Scraping Guide

This guide explains how to collect images for CORROSION and NOCORROSION into the `data/` folder.

The script `scrape_images.py` uses DuckDuckGo image search (no API key required) to find URLs, downloads them, validates with Pillow, and saves them into:

- `data/CORROSION/`
- `data/NOCORROSION/`

## Install dependencies

Create or activate your Python environment, then install the scraping requirements:

```bash
pip install -r requirements_scrape.txt
```

Note: This is separate from `requirements_gan.txt` to avoid mixing training deps with scraping deps.

## Run the scraper

Basic usage (downloads up to 300 images per class, 100 per query):

```bash
python scrape_images.py
```

Customize counts:

```bash
# Fetch up to 50 images per query, capped at 200 images per class
python scrape_images.py --per-query 50 --max-per-class 200
```

Override search queries:

```bash
python scrape_images.py \
  --corrosion "corroded metal" "rusted pipe" "corroded machinery" \
  --nocorrosion "clean metal surface" "polished stainless steel"
```

Dry run (no downloads, prints planned work):

```bash
python scrape_images.py --dry-run
```

Change output root (defaults to `data`):

```bash
python scrape_images.py --output-root data
```

## What the script enforces

- De-duplication by URL
- Content-type checks and file extensions (jpeg/png/webp/gif; jpg fallback)
- Basic image validation with Pillow (rejects corrupted images)
- Minimum image dimension (shorter side) default >= 224 px
- Polite short delay between query batches

## Tips for better datasets

- Use more diverse, specific queries to avoid near-duplicates.
- Consider adding synonyms and related contexts (e.g., "rust flakes", "pitted metal").
- Manually review a sample for label accuracy, then prune obvious mislabels.
- After scraping, you can run your existing split pipeline to distribute into train/val/test.

## Ethics and Terms of Use

- Always respect websites' terms of service and robots.txt.
- Use downloaded images only for educational or research purposes unless you have rights to use them otherwise.
- If you plan to publish a dataset, ensure proper licensing and attribution.
