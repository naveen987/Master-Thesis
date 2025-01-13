from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import json

# Set up Edge options
edge_options = Options()
edge_options.add_argument("--headless")  # Run in headless mode
edge_options.add_argument("--disable-gpu")
edge_options.add_argument("--no-sandbox")

# Path to Edge WebDriver
edge_driver_path = "C:/Tools/msedgedriver.exe"  # Update this path as needed

# Initialize WebDriver
service = Service(edge_driver_path)
driver = webdriver.Edge(service=service, options=edge_options)

try:
    # Load the main webpage
    driver.get("https://medlineplus.gov/ency/encyclopedia_A.htm")
    
    # Wait for the list of links to load
    WebDriverWait(driver, 40).until(
        EC.presence_of_element_located((By.ID, "index"))
    )
    
    # Get all relevant links from the <ul id="index">
    index_ul = driver.find_element(By.ID, "index")
    links = index_ul.find_elements(By.TAG_NAME, "a")
    links_data = [{"text": link.text, "url": link.get_attribute("href")} for link in links]
    
    # Debug: Print the number of links found
    print(f"Found {len(links_data)} links to process.")
    
    # List to store scraped data
    scraped_data = []

    # Iterate through each link and scrape content
    for index, link_data in enumerate(links_data):
        link_text = link_data["text"]
        link_url = link_data["url"]
        
        # Open the link
        print(f"Processing {index + 1}/{len(links_data)}: {link_text} ({link_url})")
        driver.get(link_url)
        
        # Wait for the article content to load
        try:
            WebDriverWait(driver, 40).until(
                EC.presence_of_element_located((By.ID, "d-article"))
            )
        except TimeoutException:
            print(f"Timeout loading content for {link_text}. Skipping...")
            continue
        
        # Get the page source
        page_source = driver.page_source
        
        # Use BeautifulSoup to parse the rendered HTML
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # Extract the title
        title = soup.find('h1').get_text(strip=True)
        
        # Extract all subsections based on headings and content
        content_div = soup.find('div', {'id': 'd-article'})
        subsections = []
        if content_div:
            current_heading = None
            for element in content_div.find_all(['h2', 'h3', 'p', 'ul']):
                if element.name in ['h2', 'h3']:
                    # Start a new subsection
                    current_heading = element.get_text(strip=True)
                    subsections.append({"heading": current_heading, "content": []})
                elif current_heading and element.name == 'p':
                    # Add paragraph to the current subsection
                    subsections[-1]["content"].append(element.get_text(strip=True))
                elif current_heading and element.name == 'ul':
                    # Add list items to the current subsection
                    list_items = [li.get_text(strip=True) for li in element.find_all('li')]
                    subsections[-1]["content"].extend(list_items)
        else:
            subsections.append({"heading": "Error", "content": ["Content not found."]})
        
        # Append to the scraped data list
        scraped_data.append({
            "title": title,
            "link_text": link_text,
            "url": link_url,
            "subsections": subsections
        })
        
        # Pause briefly to avoid overwhelming the server
        time.sleep(2)
    
    # Save the data to a JSON file
    with open("scraped_data_subsections.json", "w", encoding="utf-8") as f:
        json.dump(scraped_data, f, ensure_ascii=False, indent=4)

    print("Scraping completed. Data saved to scraped_data_subsections.json.")

finally:
    # Close the WebDriver
    driver.quit()
