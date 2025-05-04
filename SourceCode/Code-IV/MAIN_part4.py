import traceback
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from scraper_part4 import scrape_transfer_data
from processor_part4 import process_transfer_data
from config_part4 import OUTPUT_FOLDER

def run_part4():
    print("--- Part IV ---")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    driver = None
    try:
        options = ChromeOptions()
        options.add_argument('--log-level=3')
        driver = webdriver.Chrome(options=options)
        print("WebDriver ready.")

        scrape_success = scrape_transfer_data(driver)

        if scrape_success:
            print("\nProcess scraped data")
            final_df = process_transfer_data()

            if final_df is not None:
                if not final_df.empty:
                    print(f"\nProcessed data: {final_df.shape[0]} records")
                else: print("\nNo data after filter")
            else: print("\nProcess failed")

    except Exception as e:
        traceback.print_exc()
    finally:
        if driver: driver.quit()

    print("\n--- Part IV completed ---")

if __name__ == '__main__':
    run_part4()