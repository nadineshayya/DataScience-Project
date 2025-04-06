import os
import time
import re
import json
import pandas as pd
import concurrent.futures
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# Name of the JSON file that will store the discovered category URLs
CATEGORIES_FILE = 'categories.json'

# Output CSV file name
OUTPUT_CSV = 'allrecipes_limited_categories.csv'

def init_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Headless mode for speed
    options.add_argument('--disable-gpu')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--disable-quic')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def get_all_categories(driver):
    """
    Retrieve all unique AllRecipes category URLs from the main recipes page.
    """
    main_url = "https://www.allrecipes.com/recipes/"
    driver.get(main_url)
    time.sleep(2)  # Reduced wait time
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    category_links = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        # Filter only category pages
        if href.startswith("https://www.allrecipes.com/recipes/") and href != main_url:
            # Example: "https://www.allrecipes.com/recipes/76/appetizers-and-snacks/" 
            if re.search(r"/recipes/\d+/", href):
                category_links.add(href)
    return list(category_links)

def load_or_discover_categories(driver):
    """
    If categories.json exists, load categories from it.
    Otherwise, discover categories from AllRecipes and save to categories.json.
    """
    if os.path.exists(CATEGORIES_FILE):
        with open(CATEGORIES_FILE, 'r', encoding='utf-8') as f:
            category_urls = json.load(f)
        print(f"Loaded {len(category_urls)} categories from {CATEGORIES_FILE}.")
    else:
        category_urls = get_all_categories(driver)
        print(f"Discovered {len(category_urls)} categories from allrecipes.com.")
        with open(CATEGORIES_FILE, 'w', encoding='utf-8') as f:
            json.dump(category_urls, f, indent=2)
        print(f"Saved categories to {CATEGORIES_FILE}.")
    return category_urls

def get_recipe_links(page_url, driver):
    """
    Retrieve all unique recipe URLs from a given category page.
    """
    driver.get(page_url)
    time.sleep(1)  # Reduced wait time
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    recipe_links = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        if '/recipe/' in href:
            recipe_links.add(href)
    return list(recipe_links)

def scrape_recipe(url, driver):
    """
    Scrape details from a single recipe page.
    Returns a dictionary including:
      - url
      - title
      - ingredients
      - cooking_time
      - nutrition_facts
      - publish_date
      - scraped_date (timestamp)
    """
    driver.get(url)
    title = None

    # Attempt to extract title with multiple selectors
    try:
        title_tag = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'h1.headline.heading-content'))
        )
        title = title_tag.text.strip()
    except Exception:
        pass
    if not title:
        try:
            title_tag = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'h1[data-testid="recipe-title"]'))
            )
            title = title_tag.text.strip()
        except Exception:
            pass
    if not title:
        try:
            title_tag = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'h1.article-heading'))
            )
            title = title_tag.text.strip()
        except Exception:
            title = "Title not found"
            print(f"Warning: Title not found for URL: {url}")

    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Extract ingredients
    ingredients = []
    try:
        ingredient_tags = soup.find_all('li', class_='mm-recipes-structured-ingredients__list-item')
        for tag in ingredient_tags:
            quantity = tag.find('span', attrs={'data-ingredient-quantity': 'true'})
            unit = tag.find('span', attrs={'data-ingredient-unit': 'true'})
            name = tag.find('span', attrs={'data-ingredient-name': 'true'})
            parts = []
            if quantity:
                parts.append(quantity.get_text(strip=True))
            if unit:
                parts.append(unit.get_text(strip=True))
            if name:
                parts.append(name.get_text(strip=True))
            if parts:
                ingredients.append(" ".join(parts))
        if not ingredients:
            ingredients = ["Ingredients not found"]
    except Exception as e:
        print(f"Error extracting ingredients: {e}")
        ingredients = ["Ingredients not found"]

    # Extract cooking time
    cooking_time = "Not found"
    try:
        time_tag = soup.find('div', class_='mm-recipes-details__label', text='Total Time:')
        if time_tag:
            cooking_time = time_tag.find_next_sibling('div', class_='mm-recipes-details__value').get_text(strip=True)
    except Exception as e:
        print(f"Error extracting cooking time: {e}")

    # Extract nutrition facts
    nutrition_facts = {}
    try:
        nutrition_section = soup.find('div', id='mm-recipes-nutrition-facts_1-0')
        if nutrition_section:
            summary_table = nutrition_section.find('table', class_='mm-recipes-nutrition-facts-summary__table')
            if summary_table:
                rows = summary_table.find_all('tr', class_='mm-recipes-nutrition-facts-summary__table-row')
                for row in rows:
                    cells = row.find_all('td', class_='mm-recipes-nutrition-facts-summary__table-cell')
                    if len(cells) == 2:
                        key = cells[1].get_text(strip=True)
                        value = cells[0].get_text(strip=True)
                        nutrition_facts[key] = value
            detailed_table = nutrition_section.find('table', class_='mm-recipes-nutrition-facts-label__table')
            if detailed_table:
                rows = detailed_table.find_all('tr')
                for row in rows:
                    nutrient_tag = row.find('span', class_='mm-recipes-nutrition-facts-label__nutrient-name')
                    if nutrient_tag:
                        nutrient_name = nutrient_tag.get_text(strip=True)
                        value = row.get_text(strip=True).replace(nutrient_name, '').strip()
                        nutrition_facts[nutrient_name] = value
    except Exception as e:
        print(f"Error extracting nutrition facts: {e}")

    # Extract publish date (try JSON-LD first)
    publish_date = "Not found"
    try:
        script_tags = soup.find_all('script', type='application/ld+json')
        for script_tag in script_tags:
            try:
                data = json.loads(script_tag.string)
                if isinstance(data, dict) and data.get('@type', '') == 'Recipe':
                    publish_date = data.get('datePublished', 'Not found')
                    break
                elif isinstance(data, list):
                    for item in data:
                        if item.get('@type') == 'Recipe':
                            publish_date = item.get('datePublished', 'Not found')
                            break
                    if publish_date != 'Not found':
                        break
            except Exception:
                pass
    except Exception as e:
        print(f"Error extracting publish date from JSON-LD: {e}")

    if publish_date == "Not found":
        byline_elem = soup.find('div', class_='mntl-attribution__item-date')
        if byline_elem:
            publish_date = byline_elem.get_text(strip=True)

    # Ensure a standard return structure even in case of unexpected errors
    try:
        return {
            "url": url,
            "title": title or "Title not found",
            "ingredients": ingredients or ["Ingredients not found"],
            "cooking_time": cooking_time or "Unknown cooking time",
            "nutrition_facts": nutrition_facts or {},
            "publish_date": publish_date or "Unknown publish date",
            "scraped_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        print(f"Critical error in scraping {url}: {e}")
        # Guaranteed fallback:
        return {
            "url": url,
            "title": "Title not found",
            "ingredients": ["Ingredients not found"],
            "cooking_time": "Unknown cooking time",
            "nutrition_facts": {},
            "publish_date": "Unknown publish date",
            "scraped_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def scrape_category(category_url, driver):
    """
    Scrape all recipes from a given category URL by iterating through paginated pages.
    Stops when no new unique recipe links are found.
    """
    recipes = []
    seen_urls = set()
    category_name = category_url.rstrip('/').split('/')[-1]
    page = 1
    while True:
        page_url = f"{category_url}?page={page}"
        print(f"Scraping page: {page_url}")
        recipe_links = get_recipe_links(page_url, driver)
        new_links = [link for link in recipe_links if link not in seen_urls]

        if not new_links:
            print("No new unique recipes found on this page. Ending pagination for this category.")
            break

        for link in new_links:
            seen_urls.add(link)
            try:
                recipe = scrape_recipe(link, driver)
                if recipe and 'title' in recipe and recipe['title'] != "Title not found":
                    recipe['category'] = category_name
                    recipes.append(recipe)
                    print(f"Scraped: {recipe['title']} from {link}")
                else:
                    print(f"Skipped invalid recipe at {link}")
            except Exception as e:
                print(f"Error scraping {link}: {e}")
        page += 1

    return recipes

def scrape_one_category(category_url):
    driver = init_driver()
    try:
        recipes = scrape_category(category_url, driver)
        valid_recipes = [
            r for r in recipes 
            if r.get('title') and r['title'] != "Title not found" and r.get('url')
        ]
        return valid_recipes
    finally:
        driver.quit()

def clean_data(df):
    """
    Clean the DataFrame:
      - Remove rows where the title is 'Title not found' or where ingredients are missing
      - Replace publish_date 'Not found' with 'Unknown publish date'
      - Replace cooking_time 'Not found' with 'Unknown cooking time'
    """
    # Check if the DataFrame has the expected 'title' column
    if 'title' not in df.columns:
        print("Warning: 'title' column not found in DataFrame. Skipping cleaning.")
        return df

    df_clean = df[~df['title'].str.contains("Title not found", na=False)]
    
    if 'ingredients' in df_clean.columns:
        df_clean = df_clean[~df_clean['ingredients'].astype(str).str.contains("Ingredients not found")]
    
    if 'publish_date' in df_clean.columns:
        df_clean['publish_date'] = df_clean['publish_date'].replace("Not found", "Unknown publish date")
    
    if 'cooking_time' in df_clean.columns:
        df_clean['cooking_time'] = df_clean['cooking_time'].replace("Not found", "Unknown cooking time")
    
    return df_clean


def main():
    # Load existing CSV for deduplication if it exists
    try:
        df_existing = pd.read_csv(OUTPUT_CSV)
        existing_urls = set(df_existing['url'].tolist())
    except FileNotFoundError:
        df_existing = pd.DataFrame()
        existing_urls = set()

    # Load or discover categories
    driver_for_cats = init_driver()
    category_urls = load_or_discover_categories(driver_for_cats)
    driver_for_cats.quit()

    # Limit to exactly 30 categories and update the JSON file so only these 30 remain.
    category_urls = category_urls[:30]
    with open(CATEGORIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(category_urls, f, indent=2)
    print(f"Using and updating JSON with {len(category_urls)} categories.")

    print(f"Starting to scrape {len(category_urls)} categories.")

    # Scrape each category concurrently
    all_recipes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(scrape_one_category, url): url for url in category_urls}
        for future in concurrent.futures.as_completed(future_to_url):
            cat_url = future_to_url[future]
            try:
                cat_recipes = future.result()
                all_recipes.extend(cat_recipes)
                print(f"Done scraping category: {cat_url}")
            except Exception as e:
                print(f"Error scraping {cat_url}: {e}")

    # Deduplicate recipes by URL
    new_recipes = {
    recipe["url"]: recipe
    for recipe in all_recipes
    if recipe.get("url") and recipe.get("title") and recipe["title"] != "Title not found" and recipe["url"] not in existing_urls
}


    # Create DataFrame and clean data
    df_new = pd.DataFrame(list(new_recipes.values()))
    df_new = clean_data(df_new)

    # Combine with existing data and save
    if not df_new.empty:
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(OUTPUT_CSV, index=False)
        print(f"Scraping complete. {len(df_new)} new recipes saved to {OUTPUT_CSV}.")
    else:
        print("No new recipes found to add.")

if __name__ == "__main__":
    main()
