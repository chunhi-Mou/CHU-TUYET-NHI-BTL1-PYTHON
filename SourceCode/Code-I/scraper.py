from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from config_part1 import WAIT_TIME

def safe_get_text(row, data_stat):
    try:
        cell = row.find(['td', 'th'], attrs={'data-stat': data_stat})
        if not cell: return 'N/a'
        if data_stat == 'nationality':
            strings = list(cell.stripped_strings)
            if strings: return strings[-1]
            else: return 'N/a'
        else: return cell.get_text(strip=True)
    except Exception as e:
        print(f"Error get '{data_stat}': {e}")
        return 'N/a'

def scrape_fbref_data(driver: WebDriver, url_config: dict, stats_map: dict) -> dict:
    player_data = {}

    for category, (url, table_id) in url_config.items():
        print(f"Process: {category} ({url})")
        try:
            driver.get(url)
            wait = WebDriverWait(driver, WAIT_TIME)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, f"{table_id} tbody tr")))

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            rows = soup.select(f"{table_id} tbody tr")

            for row in rows:
                if 'thead' in row.get('class', []): continue

                player_name = safe_get_text(row, 'player')
                team_name = safe_get_text(row, 'team')

                if not player_name or team_name == 'N/a': continue

                key = (player_name, team_name)
                data = player_data.setdefault(key, {'Player': player_name, 'Team': team_name})

                for k, stat in stats_map.items():
                    if k not in ['Player', 'Team']:
                        val = safe_get_text(row, stat)
                        if val != 'N/a' or k not in data:
                            data[k] = val

        except TimeoutError:
            print(f"Timeout wait {table_id}")
        except Exception as e:
            print(f"Error in {category}: {e}")

    return player_data
