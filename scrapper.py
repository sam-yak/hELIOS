import time
import json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
# We no longer need webdriver_manager
# from webdriver_manager.chrome import ChromeDriverManager

MATERIALS_TO_SCRAPE = [
    "Aluminum 6061-T6",
    "ABS Plastic, General Purpose",
    "Titanium Ti-6Al-4V (Grade 5)"
]

def setup_driver():
    """
    Initializes the Selenium WebDriver using the native Selenium Manager.
    This is a more stable method.
    """
    options = webdriver.ChromeOptions()
    # You can add options to run headless (without opening a window) later
    # options.add_argument("--headless")
    
    # Selenium 4.6+ has its own driver manager. No need for a third-party library.
    driver = webdriver.Chrome(options=options)
    return driver

def scrape_material_data(driver, material_name):
    """
    Searches for a material on MatWeb, navigates to its page, and scrapes its data.
    """
    print(f"Searching for {material_name}...")
    try:
        driver.get("http://www.matweb.com/search/Search.aspx")

        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "ctl00_ContentMain_txtSearch"))
        )
        search_box.clear()
        search_box.send_keys(material_name)
        
        search_button = driver.find_element(By.ID, "ctl00_ContentMain_btnSubmit")
        search_button.click()

        first_result_link = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "#tblResults tr.datarow a"))
        )
        first_result_link.click()

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "tblMain"))
        )
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        properties = {}
        data_table = soup.find('table', id='tblMain')
        if data_table:
            rows = data_table.find_all('tr')
            current_section = "General Properties"
            for row in rows:
                header_cell = row.find('td', class_='tableheaders')
                if header_cell and header_cell.get_text(strip=True):
                    current_section = header_cell.get_text(strip=True)
                    if current_section not in properties:
                        properties[current_section] = {}
                    continue

                cells = row.find_all('td', class_='datarow')
                if len(cells) >= 2:
                    key = cells[0].get_text(strip=True)
                    value = cells[1].get_text(strip=True)
                    if key and value:
                        if current_section in properties:
                           properties[current_section][key] = value
        
        print(f"  ...found {len(properties)} sections.")
        return properties

    except Exception as e:
        print(f"An error occurred while scraping for {material_name}: {e}")
        return None

def main():
    """
    Main function to orchestrate the scraping process using Selenium.
    """
    driver = setup_driver()
    all_materials_data = {}

    for material_name in MATERIALS_TO_SCRAPE:
        scraped_properties = scrape_material_data(driver, material_name)
        if scraped_properties:
            all_materials_data[material_name] = scraped_properties
        time.sleep(2) 
    
    driver.quit()

    output_filename = 'scraped_data.json'
    with open(output_filename, 'w') as f:
        json.dump(all_materials_data, f, indent=2)
    
    print(f"\nScraping complete. Data saved to {output_filename}")

if __name__ == "__main__":
    main()
