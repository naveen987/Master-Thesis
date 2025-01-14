from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from time import sleep

# Set up Selenium WebDriver
service = Service('C:/Tools/msedgedriver.exe')  # Replace with your Edge WebDriver path
driver = webdriver.Edge(service=service)

# List of specialties to search
specialties = ["urologist", "cardiologist", "dermatologist", "orthopedist", "neurologist"]
location = "Heidelberg"  # Define the location for the search

# Initialize results list
results = []

try:
    for specialty in specialties:
        print(f"Scraping data for {specialty} in {location}...")
        
        # Open Bing Maps URL
        url = f"https://www.bing.com/maps?q={specialty}+in+{location}"
        driver.get(url)
        
        # Wait for the page to load
        wait = WebDriverWait(driver, 60)
        
        # Scroll to ensure content is loaded dynamically
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            sleep(2)  # Allow time for content to load
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
        # Locate the doctor list
        try:
            doctor_list = wait.until(EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'overlay-listings')]")))
            doctor_cards = doctor_list.find_elements(By.TAG_NAME, 'li')
            
            for card in doctor_cards:
                try:
                    # Extract Doctor Name
                    name = card.find_element(By.CLASS_NAME, 'b_factrow').text if card.find_elements(By.CLASS_NAME, 'b_factrow') else "N/A"

                    # Extract Specialty
                    specialty_text = card.find_elements(By.CLASS_NAME, 'b_factrow')[1].text if len(card.find_elements(By.CLASS_NAME, 'b_factrow')) > 1 else "N/A"

                    # Extract Address
                    address = card.find_element(By.CLASS_NAME, 'b_address').text if card.find_elements(By.CLASS_NAME, 'b_address') else "N/A"

                    # Store the result
                    results.append({
                        "Specialty": specialty,
                        "Doctor Name": name,
                        "Specialty Info": specialty_text,
                        "Address": address
                    })
                except Exception as e:
                    print(f"Error processing card: {e}")
                    continue
        
        except Exception as e:
            print(f"Error loading doctor list for {specialty}: {e}")
            continue

finally:
    # Close the browser
    driver.quit()

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv('doctors_data_multiple_specialties.csv', index=False)
print("Data successfully saved to doctors_data_multiple_specialties.csv!")
