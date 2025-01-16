from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from time import sleep

# Path to Edge WebDriver
webdriver_path = 'C:/Tools/msedgedriver.exe'  # Replace with your Edge WebDriver path

# Expanded list of specialties to search
specialties = [
    "urologist", "cardiologist", "dermatologist", "orthopedist",
    "neurologist", "endocrinologist", "gastroenterologist",
    "pulmonologist", "rheumatologist", "oncologist", "psychiatrist",
    "pediatrician", "ophthalmologist", "general surgeon",
    "vascular surgeon", "plastic surgeon", "family doctor",
    "radiologist", "otolaryngologist", "nephrologist",
    "anesthesiologist", "geriatrician"
]

# List of locations in Germany
locations = [
    "Heidelberg", "Berlin", "Munich", "Frankfurt", "Hamburg",
    "Stuttgart", "Dresden", "Leipzig", "Cologne", "Bonn",
    "Düsseldorf", "Essen", "Nuremberg", "Bremen", "Dortmund",
    "Hannover", "Mannheim", "Freiburg", "Kiel", "Wiesbaden"
]

# Initialize results list
results = []

# Function to initialize WebDriver
def initialize_driver():
    service = Service(webdriver_path)
    driver = webdriver.Edge(service=service)
    driver.maximize_window()
    return driver

try:
    for location in locations:
        for specialty in specialties:
            print(f"Scraping data for {specialty} in {location}...")
            
            # Reinitialize WebDriver for each location to ensure a fresh session
            driver = initialize_driver()
            
            try:
                # Open Bing Maps URL
                url = f"https://www.bing.com/maps?q={specialty}+in+{location}"
                driver.get(url)
                
                # Wait for the page to load
                wait = WebDriverWait(driver, 60)
                try:
                    doctor_list = wait.until(EC.presence_of_element_located((By.XPATH, "//ul[contains(@class, 'b_vlistb')]")))
                except Exception:
                    print(f"No results found for {specialty} in {location}. Skipping...")
                    continue
                
                # Scroll to ensure all content is loaded
                last_height = driver.execute_script("return document.body.scrollHeight")
                while True:
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    sleep(3)
                    new_height = driver.execute_script("return document.body.scrollHeight")
                    if new_height == last_height:
                        break
                    last_height = new_height
                
                # Locate and extract data
                try:
                    doctor_cards = doctor_list.find_elements(By.TAG_NAME, 'li')
                    for card in doctor_cards:
                        try:
                            # Extract Doctor Name
                            name = card.find_element(By.CLASS_NAME, 'b_factrow').text if card.find_elements(By.CLASS_NAME, 'b_factrow') else "N/A"

                            # Extract Specialty Info
                            specialty_text = card.find_elements(By.CLASS_NAME, 'b_factrow')[1].text if len(card.find_elements(By.CLASS_NAME, 'b_factrow')) > 1 else "N/A"

                            # Extract Address
                            address = card.find_element(By.CLASS_NAME, 'b_address').text if card.find_elements(By.CLASS_NAME, 'b_address') else "N/A"

                            # Store the result
                            results.append({
                                "Location": location,
                                "Specialty": specialty,
                                "Doctor Name": name,
                                "Specialty Info": specialty_text,
                                "Address": address
                            })
                        except Exception as e:
                            print(f"Error processing card: {e}")
                            continue
                
                except Exception as e:
                    print(f"Error loading doctor list for {specialty} in {location}: {e}")
                    continue
            
            finally:
                # Close the browser for this session
                driver.quit()

finally:
    # Save results to CSV with UTF-8 encoding
    df = pd.DataFrame(results)
    df.to_csv('raw_doctors_data_multiple_locations.csv', index=False, encoding='utf-8')
    print("Data successfully saved to doctors_data_multiple_locations.csv!")
