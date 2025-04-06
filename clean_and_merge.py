import pandas as pd
import re
import ast
from datetime import datetime
import os

def clean_url(url):
    """Trim extra whitespace from the URL; return 'not found' if missing."""
    if pd.isna(url) or not str(url).strip():
        return "not found"
    return str(url).strip()

def clean_title(title):
    """Trim extra whitespace from the title; return 'not found' if missing."""
    if pd.isna(title) or not str(title).strip():
        return "not found"
    return str(title).strip()

def clean_ingredients(ingredients):
    """
    Standardize the ingredients into a uniform semicolon-separated string.
    Remove any measurement units and special characters, keeping just the ingredient names.
    """
    if pd.isna(ingredients) or not str(ingredients).strip():
        return "not found"
    
    if isinstance(ingredients, list):
        ing_list = ingredients
    else:
        ing_str = str(ingredients).strip().strip("[]")
        if ";" in ing_str:
            ing_list = [x.strip() for x in ing_str.split(";") if x.strip()]
        else:
            ing_list = [x.strip() for x in ing_str.split(",") if x.strip()]
    
    cleaned_ingredients = []
    for ing in ing_list:
        ing = re.sub(
            r'^\d+\s*[\d/]*\s*(tbsp|tsp|cup[s]?|ounce[s]?|oz|pound[s]?|lb|g|kg|ml|l|cl|qt|pt|gal|pinch|dash|to taste|as needed)',
            '',
            ing,
            flags=re.IGNORECASE
        )
        ing = re.sub(r'^[\'"\s-]+|[\'"\s-]+$', '', ing)
        ing = re.sub(r'\([^)]*\)', '', ing)
        ing = re.sub(r'^\d+\s*[\d/]*\s*', '', ing)
        ing = ing.lower().strip()
        if ing:
            cleaned_ingredients.append(ing)
    
    return "; ".join(cleaned_ingredients) if cleaned_ingredients else "not found"

def clean_nutrition_facts(nutrition):
    """
    Standardize nutrition facts to include only Calories, Fat, Carb, and Protein.
    Extract numeric values and standardize units.
    """
    if pd.isna(nutrition) or not str(nutrition).strip() or str(nutrition).strip().lower() in ["not available", "not found"]:
        return "Calories: not found; Fat: not found; Carb: not found; Protein: not found"
    
    nutrition_dict = {
        "Calories": "not found",
        "Fat": "not found",
        "Carb": "not found",
        "Protein": "not found"
    }
    
    try:
        nf = nutrition if isinstance(nutrition, dict) else ast.literal_eval(str(nutrition))
        if isinstance(nf, dict):
            for key, value in nf.items():
                key_lower = str(key).lower()
                if "calori" in key_lower:
                    nutrition_dict["Calories"] = re.sub(r'[^\d]', '', str(value))
                elif "fat" in key_lower:
                    nutrition_dict["Fat"] = re.sub(r'[^\d]', '', str(value)) + "g"
                elif "carb" in key_lower or "carbohydrate" in key_lower:
                    nutrition_dict["Carb"] = re.sub(r'[^\d]', '', str(value)) + "g"
                elif "protein" in key_lower:
                    nutrition_dict["Protein"] = re.sub(r'[^\d]', '', str(value)) + "g"
    except:
        calories_match = re.search(r'Calories?[:\s]*([\d,]+)', str(nutrition), re.IGNORECASE)
        fat_match = re.search(r'Fat[:\s]*([\d,]+)\s*g', str(nutrition), re.IGNORECASE)
        carb_match = re.search(r'(Carbs?|Carbohydrates?)[:\s]*([\d,]+)\s*g', str(nutrition), re.IGNORECASE)
        protein_match = re.search(r'Protein[:\s]*([\d,]+)\s*g', str(nutrition), re.IGNORECASE)
        
        if calories_match:
            nutrition_dict["Calories"] = calories_match.group(1).replace(',', '')
        if fat_match:
            nutrition_dict["Fat"] = fat_match.group(1).replace(',', '') + "g"
        if carb_match:
            nutrition_dict["Carb"] = carb_match.group(2).replace(',', '') + "g"
        if protein_match:
            nutrition_dict["Protein"] = protein_match.group(1).replace(',', '') + "g"
    
    return (
        f"Calories: {nutrition_dict['Calories']}; "
        f"Fat: {nutrition_dict['Fat']}; "
        f"Carb: {nutrition_dict['Carb']}; "
        f"Protein: {nutrition_dict['Protein']}"
    )

def clean_cooking_time(time_str):
    """
    Standardize cooking time to minutes only or 'X hr Y min' format.
    """
    if pd.isna(time_str) or not str(time_str).strip():
        return "not found"
    
    time_str = str(time_str).lower().strip()
    hr_match = re.search(r'(\d+)\s*hr', time_str)
    min_match = re.search(r'(\d+)\s*min', time_str)
    
    hours = int(hr_match.group(1)) if hr_match else 0
    minutes = int(min_match.group(1)) if min_match else 0
    
    if hours > 0 and minutes > 0:
        return f"{hours} hr {minutes} min"
    elif hours > 0:
        return f"{hours} hr"
    else:
        return f"{minutes} min"

def standardize_published_date(date_str):
    """
    Convert published date to DD/MM/YYYY format.
    """
    if pd.isna(date_str) or not str(date_str).strip():
        return "not found"
    
    date_str = str(date_str).strip()
    date_str = re.sub(r'^(Published|Updated)\s*on\s*', '', date_str, flags=re.IGNORECASE)
    
    try:
        if re.match(r'\d{1,2}/\d{1,2}/\d{2,4}', date_str):
            parts = date_str.split('/')
            day = int(parts[0])
            month = int(parts[1])
            year = int("20" + parts[2]) if len(parts[2]) == 2 else int(parts[2])
            return f"{day}/{month}/{year}"
        elif re.match(r'\w+\s+\d{1,2},\s*\d{4}', date_str):
            dt = datetime.strptime(date_str, "%B %d, %Y")
            return f"{dt.day}/{dt.month}/{dt.year}"
        else:
            dt = pd.to_datetime(date_str, errors='coerce')
            if pd.notna(dt):
                return f"{dt.day}/{dt.month}/{dt.year}"
            else:
                return "not found"
    except:
        return "not found"

def standardize_scraped_date(date_str):
    """
    Standardize scraped date to YYYY-MM-DD format.
    """
    if pd.isna(date_str) or not str(date_str).strip():
        return "not found"
    
    date_str = str(date_str).strip()
    try:
        dt = pd.to_datetime(date_str, errors='coerce')
        if pd.isna(dt):
            return "not found"
        return dt.strftime("%Y-%m-%d")
    except:
        return "not found"

def clean_category(category):
    """
    Standardize category to lowercase with underscores and merge similar categories.
    
    Quick Meals Group:
      - Merge "30_minute_meals", "quick_and_easy", and "one_pot_meals" into "quick_meals_group".
      
    Desserts Group:
      - Merge "dessert", "desserts", "cake", "sweet_pies", "cupcakes_and_muffins", and "cookies" into "desserts".
      
    Additional merges:
      - Merge "dinner" into "main_dishes".
      - Merge "rice_side_dishes" into "side_dishes".
      - Merge "asian_soups" into "soup".
      - Merge "vegetables" into "fruit_veggie".
    """
    if pd.isna(category) or not str(category).strip():
        return "not found"
    
    category = str(category).strip().lower()
    category = re.sub(r'\brecipes?\b', '', category)
    category = re.sub(r'[^\w]+', '_', category)
    category = category.strip('_')
    
    mapping = {
        # Breakfast
        "breakfast_and_brunch": "breakfast_and_brunch",
        "breakfast_brunch": "breakfast_and_brunch",
        
        # Main dishes
        "main_dish": "main_dishes",
        "main_dishes": "main_dishes",
        "dinner": "main_dishes",
        
        # Desserts group
        "dessert": "desserts",
        "desserts": "desserts",
        "cake": "desserts",
        "sweet_pies": "desserts",
        "cupcakes_and_muffins": "desserts",
        "cookies": "desserts",
        
        # St. Patrick's Day
        "st_patrick_s_day": "st_patricks_day",
        "st_patricks_day": "st_patricks_day",
        
        # Pasta and noodles
        "pasta": "pasta_and_noodles",
        "noodles": "pasta_and_noodles",
        "pasta_and_noodles": "pasta_and_noodles",
        
        # Seafood
        "fish": "seafood",
        "fish_seafood": "seafood",
        "seafood": "seafood",
        
        # Salad
        "salad_main_course": "salad",
        "salad": "salad",
        
        # Side dishes
        "rice_side_dishes": "side_dishes",
        "side_dishes": "side_dishes",
        
        # Quick Meals Group
        "30_minute_meals": "quick_meals_group",
        "quick_and_easy": "quick_meals_group",
        "one_pot_meals": "quick_meals_group",
        
        # Soups
        "asian_soups": "soup",
        
        # Fruit and Vegetables
        "vegetables": "fruit_veggie",
    }
    
    return mapping.get(category, category)

def clean_csv_file(input_file, output_file):
    """
    Read the CSV file, apply all cleaning functions, and overwrite the file with the cleaned version.
    """
    try:
        df = pd.read_csv(input_file)
        if 'url' in df.columns:
            df['url'] = df['url'].apply(clean_url)
        if 'title' in df.columns:
            df['title'] = df['title'].apply(clean_title)
        if 'ingredients' in df.columns:
            df['ingredients'] = df['ingredients'].apply(clean_ingredients)
        if 'cooking_time' in df.columns:
            df['cooking_time'] = df['cooking_time'].apply(clean_cooking_time)
        if 'nutrition_facts' in df.columns:
            df['nutrition_facts'] = df['nutrition_facts'].apply(clean_nutrition_facts)
        if 'publish_date' in df.columns:
            df['publish_date'] = df['publish_date'].apply(standardize_published_date)
        if 'scraped_date' in df.columns:
            df['scraped_date'] = df['scraped_date'].apply(standardize_scraped_date)
        if 'category' in df.columns:
            df['category'] = df['category'].apply(clean_category)
        
        # Overwrite the original file with cleaned data
        df.to_csv(input_file, index=False)
        print(f"Successfully cleaned and overwritten: {input_file}")
        return input_file
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return None

def merge_cleaned_csvs(cleaned_files, output_merged):
    """
    Merge multiple cleaned CSVs into a single CSV (stacked rows) incrementally.
    If the output file already exists, append only new records (based on a unique 'url').
    Also drop low-frequency categories.
    """
    # Load new cleaned data from files
    new_dfs = []
    for file in cleaned_files:
        if file:
            df = pd.read_csv(file)
            new_dfs.append(df)
    if not new_dfs:
        print("No valid CSVs to merge.")
        return
    new_df = pd.concat(new_dfs, ignore_index=True)
    
    # Drop low-frequency categories if desired
    drop_categories = {"holidays_and_events", "everyday_cooking", "world_cuisine"}
    new_df = new_df[~new_df["category"].isin(drop_categories)]
    
    # If output file exists, load it and then append new records that are not duplicates (using 'url')
    if os.path.exists(output_merged):
        existing_df = pd.read_csv(output_merged)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        # Drop duplicates based on the unique 'url'
        combined_df.drop_duplicates(subset=["url"], inplace=True)
    else:
        combined_df = new_df
    
    combined_df.to_csv(output_merged, index=False)
    print(f"All cleaned CSVs merged into: {output_merged}")

def main():
    # Files are assumed to be in the project root (adjust paths if necessary)
    files_to_clean = [
        'allrecipes_limited_categories.csv',
        'recipetineats_limited_categories.csv',
        'spruce_eats_recipes.csv'
    ]
    
    cleaned_files = []
    for file in files_to_clean:
        cleaned = clean_csv_file(file, file)  
        cleaned_files.append(cleaned)
    
    merge_cleaned_csvs(cleaned_files, "combined_cleaned.csv")

if __name__ == "__main__":
    main()
