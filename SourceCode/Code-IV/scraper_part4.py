import os
import re
import time
import random
import pandas as pd
from bs4 import BeautifulSoup
from selenium.common import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from config_part4 import *

def safe_get(element, selector):
    if not element: return None
    try:
        el = element.select_one(selector)
        return el.get_text(strip=True) if el else None
    except: return None

def safe_get_attribute(element, selector, attr):
    if not element: return None
    try:
        el = element.select_one(selector)
        return el.get(attr) if el else None
    except: return None

def clean_value(text):
    if not text: return None
    val = re.sub(r'[^\d.]', '', text.replace(',', ''))
    if 'm' in text.lower(): return float(val) if val else None
    if 'k' in text.lower(): return float(val) / 1000 if val else None
    try: return float(val) if val else None
    except: return None

def save_data(data):
    if not data:
        print("No data to save.")
        return False
    try:
        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(RAW_DATA_FILENAME), exist_ok=True)
        df.to_csv(RAW_DATA_FILENAME, index=False, encoding='utf-8-sig')
        print(f"Saved {len(data)} records")
        return True
    except Exception as e:
        print(f"Save error: {e}")
        return False

def scrape_highest_etv(driver: WebDriver, url: str) -> float | None:
    if not url: return None
    try:
        driver.get(url)
        WebDriverWait(driver, WAIT_TIME).until(EC.presence_of_element_located((By.CSS_SELECTOR, PROFILE_PAGE_LOAD_CHECK_SELECTOR)))
        time.sleep(random.uniform(1, 2))
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        etv_text = safe_get(soup, HIGHEST_ETV_SELECTOR_PROFILE)
        return clean_value(etv_text) if etv_text else None
    except TimeoutException:
        print(f"Timeout on profile: {url.encode('ascii', errors='replace').decode('ascii')}")
    except Exception as e:
        print(f"Profile error: {url.encode('ascii', errors='replace').decode('ascii')} - {type(e).__name__}: {e}")
    return None

def scrape_transfer_data(driver: WebDriver) -> bool:
    data, scraped_keys = [], set()
    page, retries = 1, 2

    print("Scraping player list")
    try:
        driver.get(TRANSFER_URL)
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, PLAYER_TABLE_SELECTOR)))
    except Exception as e:
        print(f"Load failed: {e}")
        return False

    while True:
        print(f"Page {page}")
        for attempt in range(retries):
            try:
                WebDriverWait(driver, WAIT_TIME).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, PLAYER_ROW_SELECTOR))
                )
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                rows = soup.select(PLAYER_ROW_SELECTOR)

                if not rows:
                    time.sleep(random.uniform(1, 2))
                    driver.refresh()
                    continue

                added = 0
                for row in rows:
                    name = safe_get(row, PLAYER_NAME_SELECTOR)
                    team = safe_get(row, TEAM_NAME_SELECTOR)
                    url = safe_get_attribute(row, PLAYER_NAME_SELECTOR, 'href')
                    if not (name and team and url): continue

                    key = (name.strip(), team.strip())
                    if key in scraped_keys: continue

                    player = {
                        'Player': key[0],
                        'Team_TransferSite': key[1],
                        'Age': safe_get(row, AGE_SELECTOR),
                        'Position': safe_get(row, POSITION_SELECTOR),
                        TARGET_VARIABLE: clean_value(safe_get(row, ETV_SELECTOR)),
                        'Profile_URL': url,
                        'Highest_ETV': None
                    }
                    data.append(player)
                    scraped_keys.add(key)
                    added += 1

                print(f"+{added} players")
                break
            except TimeoutException:
                print(f"Timeout on page {page}, attempt {attempt+1}")
                time.sleep(random.uniform(2, 4))
            except Exception as e:
                print(f"Error on page {page}, attempt {attempt+1}: {e}")
                time.sleep(random.uniform(1, 3))

        else:
            print(f"Failed page {page} after {retries} attempts.")
            break

        try:
            next_btn = WebDriverWait(driver, WAIT_TIME).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, NEXT_PAGE_SELECTOR))
            )
            driver.execute_script("arguments[0].click();", next_btn)
            time.sleep(random.uniform(2, 4))
            page += 1
        except:
            break

    print(f"Collected {len(data)} players.")

    print("Profile ETVs")
    for i, player in enumerate(data):
        print(f"{i+1}/{len(data)}: {player['Player'].encode('ascii', errors='replace').decode('ascii')}")
        url = player.get('Profile_URL')
        player['Highest_ETV'] = scrape_highest_etv(driver, url) if url else None
        time.sleep(random.uniform(0.5, 1.5))

    for p in data:
        p.pop('Profile_URL', None)

    return save_data(data)
