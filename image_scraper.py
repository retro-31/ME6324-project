import os
import io
import time
import hashlib
import requests
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options


# ============================================================
# Chrome / Chromium Setup
# ============================================================
def get_chrome_driver():
    chrome_path = "/home/ashish/chrome_stable/chrome-linux/chrome"
    driver_path = "/home/ashish/chrome_stable/chromedriver-linux64/chromedriver"


    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # ✅ Use your manually downloaded Chromium binary
    chrome_options.binary_location = chrome_path

    # ✅ Always use the latest driver available
    service = Service(driver_path)

    print("Using Chromium binary:", chrome_path)
    print("Using ChromeDriver:", driver_path)

    return webdriver.Chrome(service=service, options=chrome_options)


# ============================================================
# Image Scraping Helpers
# ============================================================
def fetch_image_urls(query: str, max_links_to_fetch: int, wd: webdriver.Chrome, sleep_between_interactions: int = 1):
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)

    search_url = f"https://www.google.com/search?tbm=isch&q={query}"
    wd.get(search_url)

    image_urls = set()
    results_start = 0

    while len(image_urls) < max_links_to_fetch:
        scroll_to_end(wd)
        thumbnail_results = wd.find_elements("css selector", "img.Q4LuWd")
        number_results = len(thumbnail_results)

        print(f"Found {number_results} thumbnails, extracting links {results_start}:{number_results}...")

        for img in thumbnail_results[results_start:number_results]:
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            actual_images = wd.find_elements("css selector", "img.n3VNCb")
            for actual_image in actual_images:
                src = actual_image.get_attribute("src")
                if src and "http" in src:
                    image_urls.add(src)

            if len(image_urls) >= max_links_to_fetch:
                print(f"✅ Found {len(image_urls)} image links for '{query}'")
                break

        results_start = len(thumbnail_results)
        print(f"Currently have {len(image_urls)} images...")

    return image_urls


def persist_image(folder_path: str, url: str):
    try:
        image_content = requests.get(url, timeout=10).content
    except Exception as e:
        print(f"⚠️ ERROR - Could not download {url} - {e}")
        return

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert("RGB")
        file_path = os.path.join(folder_path, hashlib.sha1(image_content).hexdigest()[:10] + ".jpg")
        with open(file_path, "wb") as f:
            image.save(f, "JPEG", quality=85)
        print(f"✅ Saved: {file_path}")
    except Exception as e:
        print(f"⚠️ ERROR - Could not save {url} - {e}")


def search_and_download(search_term: str, target_path: str, number_images=100):
    target_folder = os.path.join(target_path, "_".join(search_term.lower().split(" ")))
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with get_chrome_driver() as wd:
        res = fetch_image_urls(search_term, number_images, wd=wd, sleep_between_interactions=0.5)

    for elem in res:
        persist_image(target_folder, elem)


# ============================================================
# Main Entry Point
# ============================================================
if __name__ == "__main__":
    corrosion_path = "/data/ashish/SAITEJAMS/AI_IN_MAN_Project/Corrossion"
    non_corrosion_path = "/data/ashish/SAITEJAMS/AI_IN_MAN_Project/Non_Corrosion"

    os.makedirs(corrosion_path, exist_ok=True)
    os.makedirs(non_corrosion_path, exist_ok=True)

    print("🔹 Downloading corrosion images...")
    search_and_download("corroded steel surface", corrosion_path, number_images=200)

    print("🔹 Downloading non-corrosion images...")
    search_and_download("clean steel surface", non_corrosion_path, number_images=200)

    print("✅ Done! All images saved.")
