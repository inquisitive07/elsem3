from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import csv
from pathlib import Path


def setup_chrome_driver():
    """Configure and return optimized Chrome driver"""
    options = Options()
    
    # Performance optimizations
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-logging")
    options.add_argument("--log-level=3")
    options.add_argument("--silent")
    
    # Privacy & stealth
    options.add_argument("--mute-audio")
    options.add_argument("--disable-notifications")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
    options.add_experimental_option("useAutomationExtension", False)
    
    # Reduce resource usage
    options.add_argument("--disable-images")  # Don't load images
    prefs = {
        "profile.managed_default_content_settings.images": 2,  # Block images
        "profile.default_content_setting_values.notifications": 2  # Block notifications
    }
    options.add_experimental_option("prefs", prefs)
    
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )


def extract_description(driver):
    """Extract video description with optimized error handling"""
    try:
        # Scroll to description area
        driver.execute_script("window.scrollBy(0, 300);")
        time.sleep(0.8)  # Reduced from 1s
        
        # Try to expand description
        try:
            expand_button = driver.find_element(By.CSS_SELECTOR, "tp-yt-paper-button#expand")
            driver.execute_script("arguments[0].click();", expand_button)
            time.sleep(0.3)  # Reduced from 0.5s
        except NoSuchElementException:
            pass  # Button not found, description might already be expanded
        
        # Extract description text
        desc_element = driver.find_element(
            By.CSS_SELECTOR,
            "ytd-text-inline-expander#description-inline-expander yt-attributed-string"
        )
        return desc_element.text.strip()
        
    except (NoSuchElementException, TimeoutException):
        return ""
    except Exception as e:
        print(f"⚠️ Description extraction failed: {e}")
        return ""


def get_recommendation_candidates(driver, current_url):
    """Scroll and extract recommendation links"""
    # Scroll to load recommendations
    for _ in range(2):
        driver.execute_script("window.scrollBy(0, 800);")
        time.sleep(0.8)  # Reduced from 1s
    
    # Get all links at once
    all_links = driver.find_elements(By.XPATH, "//a[@href]")
    candidates = []
    
    for link in all_links:
        try:
            href = link.get_attribute("href")
            
            # Fast validation
            if not href or "watch?v=" not in href or href == current_url:
                continue
            
            # Get title from any available attribute
            title = (
                link.get_attribute("title") or 
                link.get_attribute("aria-label") or 
                link.text
            )
            
            # Validate title length
            if title and len(title.strip()) >= 15:
                candidates.append((title.strip(), href))
                
        except Exception:
            continue  # Skip problematic links silently
    
    return candidates


def scrape_recommendations(seed_url, max_steps=50):
    """
    Main scraper function - crawls YouTube recommendations
    
    Same logic as original, optimized for:
    - Faster page loads (reduced waits)
    - Lower memory usage (disabled images)
    - Better error handling
    - Cleaner code structure
    """
    # Setup paths
    BASE_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = BASE_DIR / "main" / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE = DATA_DIR / "recommendation_walk.csv"
    
    # Initialize
    print("🚀 Starting optimized scraper in HEADLESS mode...")
    print(f"📍 Seed URL: {seed_url}")
    print(f"📁 Output file: {OUTPUT_FILE}")
    print(f"📊 Max steps: {max_steps}\n")
    
    driver = setup_chrome_driver()
    wait = WebDriverWait(driver, 30)
    
    # Create CSV with headers
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["step", "title", "url", "description"])
    
    # Load initial page
    driver.get(seed_url)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "ytd-app")))
    time.sleep(3)  # Reduced from 4s
    
    current_url = seed_url
    
    # Main crawl loop
    for step in range(max_steps):
        print(f"\n{'='*60}")
        print(f"STEP {step}/{max_steps}")
        print(f"{'='*60}")
        
        # Extract description
        description = extract_description(driver)
        print(f"📝 Description: {len(description)} chars")
        
        # Get recommendation candidates
        candidates = get_recommendation_candidates(driver, current_url)
        
        if not candidates:
            print("❌ No recommendations found. Ending walk.")
            break
        
        # Select first recommendation (same logic as original)
        title, next_url = candidates[0]
        print(f"📺 TITLE: {title}")
        
        # Save to CSV
        with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([step, title, next_url, description])
        
        # Navigate to next video
        current_url = next_url
        driver.get(next_url)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "ytd-app")))
        time.sleep(2.5)  # Reduced from 3s
    
    # Cleanup
    driver.quit()
    print(f"\n✅ SCRAPING COMPLETE")
    print(f"📁 Saved to: {OUTPUT_FILE}\n")


if __name__ == "__main__":
    import sys
    seed_url = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=8nHBGFKLHZQ"
    scrape_recommendations(seed_url)