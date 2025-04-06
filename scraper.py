import os
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

CATEGORIES_FILE = 'spruce_eats_categories.json'
OUTPUT_CSV = 'spruce_eats_recipes.csv'
MAX_CATEGORIES = 18 

def init_driver():
    options = webdriver.ChromeOptions()
    # --- Edit: Set page load strategy to "eager" to avoid waiting for full load ---
    options.page_load_strategy = "eager"
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("--enable-unsafe-swiftshader")
    options.add_argument('--log-level=3')
    options.add_argument('--disable-logging')
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.stylesheets": 2
    }
    options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    # --- Edit: Set a lower page load timeout (30 seconds) ---
    driver.set_page_load_timeout(30)
    return driver

def get_all_categories(driver):
    main_url = "https://www.thespruceeats.com/"
    driver.get(main_url)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "global-nav__list"))
        )
    except Exception as e:
        print(f"Error waiting for main page: {e}")
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    nav_list = soup.find('ul', class_='global-nav__list')
    if not nav_list:
        print("Navigation list not found!")
        return []
    
    categories = []
    target_categories = ['Recipes', 'By Region', 'Ingredients', 'Occasions']
    
    for li in nav_list.find_all('li', class_='global-nav__list-item'):
        a_tag = li.find('a', class_='global-nav__list-item-link')
        if not a_tag:
            continue
        category_name = a_tag.get_text(strip=True)
        if category_name not in target_categories:
            continue
        sub_menu = li.find('ul', class_='global-nav__sub-list')
        if not sub_menu:
            continue
        for sub_li in sub_menu.find_all('li', class_='global-nav__sub-list-item'):
            sub_a = sub_li.find('a', class_='global-nav__sub-list-item-link')
            if sub_a and 'View all' not in sub_a.get_text(strip=True):
                categories.append({
                    'name': sub_a.get_text(strip=True),
                    'url': sub_a['href'],
                    'parent_category': category_name
                })
    return categories

def filter_and_limit_categories(categories):
    filtered = []
    for cat in categories:
        filtered.append(cat)
    return filtered

def load_or_discover_categories(driver):
    if os.path.exists(CATEGORIES_FILE):
        with open(CATEGORIES_FILE, 'r', encoding='utf-8') as f:
            categories = json.load(f)
        print(f"Loaded {len(categories)} categories from {CATEGORIES_FILE}.")
    else:
        categories = get_all_categories(driver)
        categories = filter_and_limit_categories(categories)
        with open(CATEGORIES_FILE, 'w', encoding='utf-8') as f:
            json.dump(categories, f, indent=2)
        print(f"Discovered and saved {len(categories)} category URLs to {CATEGORIES_FILE}.")
    return categories

def get_recipe_links(page_url, driver):
    driver.get(page_url)
    try:
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "a.mntl-card-list-items"))
        )
    except Exception as e:
        print(f"Timeout waiting for recipes on {page_url}: {e}")
        return []
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    recipe_links = set()
    
    cards = soup.find_all('a', class_='mntl-card-list-items')
    for card in cards:
        href = card.get('href')
        if href and href.startswith('https://www.thespruceeats.com/') and 'View all' not in card.get_text():
            recipe_links.add(href)
    
    return list(recipe_links)

def scrape_recipe(url, driver, retries=2):
    attempt = 0
    while attempt < retries:
        try:
            driver.get(url)
        except Exception as e:
            print(f"Error loading page {url}: {e} (attempt {attempt+1}/{retries})")
            try:
                driver.execute_script("window.stop();")
            except Exception as stop_e:
                print(f"Error stopping page load for {url}: {stop_e}")
        try:
            # --- Edit: Lower WebDriverWait to 10 seconds ---
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "h1.heading__title"))
            )
            break
        except Exception as e:
            attempt += 1
            print(f"Timeout waiting for title on {url}: {e} (attempt {attempt}/{retries})")
            if attempt >= retries:
                print(f"Skipping {url} after {attempt} attempts due to timeout.")
                return None

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    title_tag = soup.find('h1', class_='heading__title')
    title = title_tag.get_text(strip=True) if title_tag else "Title not found"
    
    publish_date = "Not found"
    time_tag = soup.find('time')
    if time_tag:
        publish_date = time_tag.get('datetime', time_tag.get_text(strip=True))
    else:
        date_div = soup.find('div', class_='mntl-attribution__item-date')
        if date_div:
            publish_date = date_div.get_text(strip=True)
    
    cooking_time = "Not specified"
    total_time_tag = soup.find('div', class_='total-time')
    if total_time_tag:
        cooking_time = total_time_tag.get_text(strip=True).replace('Total:', '').strip()
    
    ingredients = []
    ingredients_section = soup.find('section', class_='section--ingredients')
    if ingredients_section:
        for li in ingredients_section.find_all('li', class_='structured-ingredients__list-item'):
            ingredient_parts = []
            for span in li.find_all('span'):
                if span.has_attr('data-ingredient-quantity'):
                    ingredient_parts.append(span.get_text(strip=True))
                elif span.has_attr('data-ingredient-unit'):
                    ingredient_parts.append(span.get_text(strip=True))
                elif span.has_attr('data-ingredient-name'):
                    ingredient_parts.append(span.get_text(strip=True))
            if ingredient_parts:
                ingredients.append(' '.join(ingredient_parts))
    
    nutrition_facts = {}
    nutrition_section = soup.find('div', id=re.compile("nutrition-info"))
    if nutrition_section:
        table = nutrition_section.find('table', class_='nutrition-info__table')
        if table:
            for row in table.find_all('tr', class_='nutrition-info__table--row'):
                cells = row.find_all('td', class_='nutrition-info__table--cell')
                if len(cells) == 2:
                    nutrient_name = cells[1].get_text(strip=True)
                    nutrient_value = cells[0].get_text(strip=True)
                    nutrition_facts[nutrient_name] = nutrient_value
    nutrition_str = ", ".join([f"{k}: {v}" for k, v in nutrition_facts.items()]) if nutrition_facts else "Not available"
    
    scraped_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        "url": url,
        "title": title,
        "ingredients": ingredients,
        "cooking_time": cooking_time,
        "nutrition_facts": nutrition_str,
        "publish_date": publish_date,
        "scraped_date": scraped_date
    }

def scrape_category(category_info, driver):
    recipes = []
    seen_urls = set()
    category_name = category_info['name']
    # Note: We no longer insert parent_category into the recipe data.
    category_url = category_info['url']
    
    print(f"Scraping category: {category_name} ({category_info['parent_category']})")
    print(f"Scraping page: {category_url}")
    recipe_links = get_recipe_links(category_url, driver)
    new_links = [link for link in recipe_links if link not in seen_urls]
    
    if not new_links:
        print("No new recipes found on this page.")
        return recipes
                
    for link in new_links:
        seen_urls.add(link)
        try:
            data = scrape_recipe(link, driver)
            if data and "title" in data and data["title"] != "Title not found":
                data['category'] = category_name
                recipes.append(data)
                print(f"Scraped: {data['title']}")
        except Exception as e:
            print(f"Error scraping {link}: {e}")
    
    return recipes

def scrape_one_category(category_info):
    driver = init_driver()
    try:
        return scrape_category(category_info, driver)
    finally:
        driver.quit()

def clean_data(df):
    if 'title' not in df.columns:
        print("No title column found in DataFrame!")
        return df
    df_clean = df[~df['title'].str.contains("Title not found", na=False)]
    df_clean = df_clean[df_clean['ingredients'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]
    return df_clean

def main():
    try:
        df_existing = pd.read_csv(OUTPUT_CSV)
        existing_urls = set(df_existing['url'].tolist())
    except FileNotFoundError:
        df_existing = pd.DataFrame()
        existing_urls = set()

    driver_for_cats = init_driver()
    categories = load_or_discover_categories(driver_for_cats)
    driver_for_cats.quit()

    categories = categories[:MAX_CATEGORIES]
    print(f"Total categories to scrape after limiting: {len(categories)}")
    
    all_recipes = []
    print(f"Starting to scrape {len(categories)} categories.")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_cat = {executor.submit(scrape_one_category, cat): cat for cat in categories}
        for future in concurrent.futures.as_completed(future_to_cat):
            cat_info = future_to_cat[future]
            try:
                cat_recipes = future.result()
                all_recipes.extend(cat_recipes)
                print(f"Done scraping category: {cat_info['name']} ({len(cat_recipes)} recipes)")
            except Exception as e:
                print(f"Error scraping {cat_info['name']}: {e}")

    new_recipes = {
        r["url"]: r 
        for r in all_recipes 
        if r is not None 
        and "title" in r 
        and r["title"] != "Title not found" 
        and r["url"] not in existing_urls
    }
    
    df_new = pd.DataFrame(list(new_recipes.values()))
    df_new = clean_data(df_new)
    
    if not df_new.empty:
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(OUTPUT_CSV, index=False)
        print(f"Scraping complete. {len(df_new)} new recipes saved to {OUTPUT_CSV}.")
    else:
        print("No new recipes found to add.")

if __name__ == "__main__":
    main()
