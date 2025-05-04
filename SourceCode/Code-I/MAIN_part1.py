import traceback
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from processor import process_data
from scraper import scrape_fbref_data
from config_part1 import*

def run_scraper():
    print("--- Start Code for I ---")
    driver = None
    try:
        # 1. Init Selenium WebDriver
        options = Options()
        options.add_argument('--log-level=3')
        driver = webdriver.Chrome(options=options)
        print("WebDriver started")
        # 2. Process data
        raw_data = scrape_fbref_data(driver, URL_CONFIG, STATS_MAP)
        print(f"\nScraped data for {len(raw_data)} players.")
        if not raw_data: return print("No data scraped")

        df = process_data(raw_data, list(STATS_MAP))
        if df.empty: return print("Processed data is empty")
        # 3. Save to csv
        df.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8-sig')
        print(f"Saved data for {len(df)} players.")
    except Exception:
        traceback.print_exc()
    finally:
        if driver:
            driver.quit()
            print("WebDriver closed")
    print("--- Code I finished ---")

if __name__ == '__main__':
    run_scraper()
