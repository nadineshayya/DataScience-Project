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

CATEGORIES_FILE = 'categories1.json'
OUTPUT_CSV = 'recipetineats_limited_categories.csv'

def init_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("--enable-unsafe-swiftshader")
    # Suppress Chrome logs
    options.add_argument('--log-level=3')
    options.add_argument('--disable-logging')
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
    # Disable images and CSS to speed up page loading
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.stylesheets": 2
    }
    options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def get_all_categories(driver):
    main_url = "https://www.recipetineats.com/"
    driver.get(main_url)
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    except Exception as e:
        print(f"Error waiting for main page: {e}")
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    cat_link = soup.find('a', href=re.compile(r"https://www\.recipetineats\.com/categories/"))
    if not cat_link:
        print("Recipes By Category link not found!")
        return {}
    cat_li = cat_link.find_parent("li")
    if not cat_li:
        print("Parent <li> for Recipes By Category not found!")
        return {}
    categories_container = cat_li.find("ul", class_="sub-menu")
    if not categories_container:
        print("Categories container not found!")
        return {}

    def parse_menu(ul):
        menu = {}
        for li in ul.find_all("li", recursive=False):
            a_tag = li.find("a", href=True)
            if a_tag:
                cat_name = a_tag.get_text(strip=True)
                cat_url = a_tag["href"]
                sub_ul = li.find("ul", class_="sub-menu")
                if sub_ul:
                    menu[cat_name] = {"url": cat_url, "subcategories": parse_menu(sub_ul)}
                else:
                    menu[cat_name] = {"url": cat_url}
        return menu

    categories = parse_menu(categories_container)
    return categories

def count_categories(categories):
    total = 0
    for data in categories.values():
        total += 1
        if "subcategories" in data:
            total += count_categories(data["subcategories"])
    return total

def load_or_discover_categories(driver):
    if os.path.exists(CATEGORIES_FILE):
        with open(CATEGORIES_FILE, 'r', encoding='utf-8') as f:
            categories = json.load(f)
        print(f"Loaded {len(categories)} categories from {CATEGORIES_FILE}.")
    else:
        nested_cats = get_all_categories(driver)
        total_nested = count_categories(nested_cats)
        print(f"Discovered a nested structure with {total_nested} categories.")

        def flatten_categories(cat_dict):
            urls = []
            for data in cat_dict.values():
                urls.append(data["url"])
                if "subcategories" in data:
                    urls.extend(flatten_categories(data["subcategories"]))
            return urls

        categories = flatten_categories(nested_cats)
        with open(CATEGORIES_FILE, 'w', encoding='utf-8') as f:
            json.dump(categories, f, indent=2)
        print(f"Discovered and saved {len(categories)} category URLs to {CATEGORIES_FILE}.")
    return categories

def get_recipe_links(page_url, driver):
    driver.get(page_url)
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    except Exception as e:
        print(f"Timeout or error waiting for page {page_url}: {e}")
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    recipe_links = set()
    # Define keywords that indicate non-recipe pages.
    exclusion_keywords = ['contact', 'policy', 'privacy', 'about', 'terms', 'disclaimer']
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.startswith("https://www.recipetineats.com/") and "/category/" not in href:
            # Skip links containing any excluded keyword.
            if any(keyword in href.lower() for keyword in exclusion_keywords):
                continue
            recipe_links.add(href)
    return list(recipe_links)
def scrape_recipe(url, driver, retries=2):
    attempt = 0
    while attempt < retries:
        driver.get(url)
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "h1.entry-title"))
            )
            break
        except Exception as e:
            attempt += 1
            print(f"Timeout waiting for title on {url}: {e} (attempt {attempt}/{retries})")
            if attempt >= retries:
                print(f"Skipping {url} after {attempt} attempts due to timeout.")
                return None

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    title_tag = soup.find("h1", class_="entry-title")
    title = title_tag.get_text(strip=True) if title_tag else "Title not found"

    publish_time_tag = soup.find("time", class_="entry-time")
    if publish_time_tag:
        publish_date = publish_time_tag.get_text(strip=True).replace("Published:", "").strip()
    else:
        publish_date = "Not found"

    def _extract_minutes(container_class):
        container = soup.find("div", class_=container_class)
        if not container:
            return 0
        time_element = container.find("span", class_="wprm-recipe-time")
        if not time_element:
            return 0
        hours = 0
        hours_span = time_element.find("span", class_="wprm-recipe-details-hours")
        if hours_span:
            try:
                hours = int(''.join(filter(str.isdigit, hours_span.get_text(strip=True))))
            except ValueError:
                pass
        minutes = 0
        minutes_span = time_element.find("span", class_="wprm-recipe-details-minutes")
        if minutes_span:
            try:
                minutes = int(''.join(filter(str.isdigit, minutes_span.get_text(strip=True))))
            except ValueError:
                pass
        return hours * 60 + minutes

    prep_minutes = _extract_minutes("wprm-recipe-prep-time-container")
    cook_minutes = _extract_minutes("wprm-recipe-cook-time-container")
    total_minutes = prep_minutes + cook_minutes

    cooking_time = "Not specified"
    if total_minutes > 0:
        hours = total_minutes // 60
        minutes = total_minutes % 60
        parts = []
        if hours > 0:
            parts.append(f"{hours} hr")
        if minutes > 0:
            parts.append(f"{minutes} min")
        cooking_time = " ".join(parts) if parts else "0 min"

    ingredients = []
    ing_container = soup.find("div", class_=re.compile("wprm-recipe-ingredients-container"))
    if ing_container:
        li_tags = ing_container.find_all("li", class_="wprm-recipe-ingredient")
        for li in li_tags:
            parts = []
            amount = li.find("span", class_="wprm-recipe-ingredient-amount")
            unit = li.find("span", class_="wprm-recipe-ingredient-unit")
            name = li.find("span", class_="wprm-recipe-ingredient-name")
            if amount:
                parts.append(amount.get_text(strip=True))
            if unit:
                parts.append(unit.get_text(strip=True))
            if name:
                parts.append(name.get_text(strip=True))
            if parts:
                ingredients.append(" ".join(parts))
    if not ingredients:
        ingredients = ["Ingredients not found"]

    # Extract nutrition facts
    nutrition_data = {}
    nutrition_container = soup.find("div", class_="wprm-nutrition-label-container")
    if nutrition_container:
        nutrition_items = nutrition_container.find_all("span", class_="wprm-nutrition-label-text-nutrition-container")
        for item in nutrition_items:
            label = item.find("span", class_="wprm-nutrition-label-text-nutrition-label")
            value = item.find("span", class_="wprm-nutrition-label-text-nutrition-value")
            unit = item.find("span", class_="wprm-nutrition-label-text-nutrition-unit")
            if label and value:
                label_text = label.get_text(strip=True).replace(":", "")
                value_text = value.get_text(strip=True)
                if unit:
                    value_text += unit.get_text(strip=True)
                nutrition_data[label_text] = value_text
    
    # Format nutrition facts as a string
    nutrition_facts = ", ".join([f"{k}: {v}" for k, v in nutrition_data.items()]) if nutrition_data else "Not available"

    scraped_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    category = url.split('/')[-2] if '/' in url.rstrip('/') else "Unknown"

    return {
        "url": url,
        "title": title,
        "ingredients": ingredients,
        "cooking_time": cooking_time,
        "nutrition_facts": nutrition_facts,
        "publish_date": publish_date,
        "scraped_date": scraped_date,
        "category": category
    }
def scrape_category(category_url, driver):
    recipes = []
    seen_urls = set()
    category_name = category_url.rstrip('/').split('/')[-1]
    page = 1
    max_pages = 2  # Adjust this if you want more pages per category
    while page <= max_pages:
        page_url = category_url if page == 1 else f"{category_url.rstrip('/')}/page/{page}/"
        print(f"Scraping page: {page_url}")
        recipe_links = get_recipe_links(page_url, driver)
        new_links = [link for link in recipe_links if link not in seen_urls]
        if not new_links:
            print("No new unique recipes found. Ending pagination.")
            break
        for link in new_links:
            seen_urls.add(link)
            try:
                data = scrape_recipe(link, driver)
                if data:
                    data["category"] = category_name
                    recipes.append(data)
                    print(f"Scraped: {data['title']} from {link}")
            except Exception as e:
                print(f"Error scraping {link}: {e}")
        page += 1
    return recipes

def scrape_one_category(category_url):
    driver = init_driver()
    try:
        return scrape_category(category_url, driver)
    finally:
        driver.quit()

def clean_data(df):
    df_clean = df[~df['title'].str.contains("Title not found", na=False)]
    df_clean = df_clean[~df_clean['ingredients'].astype(str).str.contains("Ingredients not found")]
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

    # For demo: limit to first 22 categories
    limited_category_urls = categories[:22]
    with open(CATEGORIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(limited_category_urls, f, indent=2)

    all_recipes = []
    print(f"Starting to scrape {len(limited_category_urls)} categories.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_to_url = {executor.submit(scrape_one_category, url): url for url in limited_category_urls}
        for future in concurrent.futures.as_completed(future_to_url):
            cat_url = future_to_url[future]
            try:
                cat_recipes = future.result()
                all_recipes.extend(cat_recipes)
                print(f"Done scraping category: {cat_url}")
            except Exception as e:
                print(f"Error scraping {cat_url}: {e}")

    new_recipes = {r["url"]: r for r in all_recipes if r["url"] not in existing_urls}
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
