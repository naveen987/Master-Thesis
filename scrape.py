from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# Set up Edge options
edge_options = Options()
edge_options.add_argument("--headless")  # Run in headless mode (no browser window)

# Path to Edge WebDriver
edge_driver_path = "C:/Tools/msedgedriver.exe"  # Update this path as needed

# Initialize WebDriver
service = Service(edge_driver_path)
driver = webdriver.Edge(service=service, options=edge_options)

try:
    # Load the webpage
    driver.get("https://medlineplus.gov/ency/patientinstructions/000382.htm")
    
    # Wait for the main article content to load
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.ID, "d-article"))
    )
    
    # Get the page source after the content loads
    page_source = driver.page_source
    
    # Use BeautifulSoup to parse the rendered HTML
    soup = BeautifulSoup(page_source, 'html.parser')
    
    # Extract the title
    title = soup.find('h1').get_text(strip=True)
    
    # Extract the article content
    content_div = soup.find('div', {'id': 'd-article'})
    if content_div:
        paragraphs = content_div.find_all('p')
        content = "\n\n".join([p.get_text(strip=True) for p in paragraphs])
    else:
        content = "Content not found."

    # Print the scraped data
    print(f"Title: {title}\n")
    print("Content:")
    print(content)

finally:
    # Close the WebDriver
    driver.quit()
