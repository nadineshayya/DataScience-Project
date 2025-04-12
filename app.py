from __future__ import annotations

import re
import random
import itertools
from pathlib import Path
from functools import lru_cache

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px 
import pulp
import streamlit as st
from dateutil import parser

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from scipy.optimize import minimize

st.set_page_config(page_title="🍽 Recipe Insight Hub", layout="wide")
DATA_PATH = Path("combined_cleaned.csv")

@st.cache_data(show_spinner="Loading recipes …")
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("Original columns:", df.columns.tolist())  # Debug print
    
    # Parse dates
    df["publish_date"] = df["publish_date"].apply(lambda x: _parse_date(str(x)))
    df["cook_min"] = df["cooking_time"].apply(_to_minutes)
    
    # Parse nutrition facts
    nutrition_cols = ["calories", "fat_g", "carb_g", "protein_g"]
    df[nutrition_cols] = df["nutrition_facts"].apply(_parse_nutrition)
    print("After parsing nutrition:", df.columns.tolist())  # Debug print
    
    df["ing_list"] = df["ingredients"].str.split(r"\s*;\s*")
    return df

# Helper functions
def _parse_date(s: str):
    try:
        return parser.parse(s, dayfirst=True)
    except Exception:
        return pd.NaT

def _to_minutes(t: str) -> float:
    if pd.isna(t):
        return np.nan
    total = 0
    hrs = re.search(r"(\d+)\s*hr", t)
    if hrs:
        total += int(hrs.group(1)) * 60
    mins = re.search(r"(\d+)\s*min", t)
    if mins:
        total += int(mins.group(1))
    return total or np.nan

def _parse_nutrition(s: str):
    nums = re.findall(r"(\d+)", s)
    # Ensure exactly four values, pad with NaN if necessary
    return pd.Series([int(x) for x in nums[:4]] + [np.nan] * (4 - len(nums)))

@lru_cache(maxsize=None)
def keyword_bank():
    return {
        "Produce": ["tomato", "onion", "pepper", "garlic", "carrot", "spinach", "apple", "lemon", "lime", "potato"],
        "Protein": ["chicken", "beef", "pork", "shrimp", "salmon", "egg", "tofu", "lentil", "bean", "turkey"],
        "Carbs": ["rice", "pasta", "noodle", "bread", "quinoa", "tortilla", "flour"],
        "Dairy": ["milk", "cheese", "butter", "yogurt", "cream", "parmesan"],
        "Spices": ["cumin", "coriander", "turmeric", "oregano", "basil", "chili", "paprika", "thyme"],
        "Condiments": ["soy sauce", "fish sauce", "ketchup", "mustard", "vinegar", "honey", "sauce"],
    }

def classify_ing(ing: str) -> str:
    low = ing.lower()
    for grp, words in keyword_bank().items():
        if any(w in low for w in words):
            return grp
    return "Other"
### 1. Add these functions to your ML utilities section ###

@st.cache_resource
def load_time_prediction_model():
    """Train and cache a cooking time prediction model"""
    from sklearn.ensemble import RandomForestRegressor
    
    # Prepare data
    df['ing_count'] = df['ing_list'].apply(len)
    df['ing_length'] = df['ingredients'].str.len()
    time_df = df.dropna(subset=['cook_min', 'ing_count', 'ing_length'])
    
    X = time_df[['ing_count', 'ing_length']]
    y = time_df['cook_min']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def plot_cooking_tips(predicted_mins):
    """Show visual tips based on predicted time"""
    import plotly.express as px
    
    tips = {
        "Quick Prep (<30min)": ["Chop veggies in advance", "Use pre-cooked proteins"],
        "Medium (30-60min)": ["Multi-task steps", "Prep while cooking"],
        "Long (>60min)": ["Use slow cooker", "Make double portions"]
    }
    
    if predicted_mins < 30:
        selected = "Quick Prep (<30min)"
    elif predicted_mins < 60:
        selected = "Medium (30-60min)"
    else:
        selected = "Long (>60min)"
    
    fig = px.bar(
        x=list(tips.keys()),
        y=[1, 1, 1],
        color_discrete_sequence=['lightgray']*3,
        title="Time-Saving Tips"
    )
    fig.update_traces(marker_color=['gold' if k == selected else 'lightgray' for k in tips.keys()])
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    
    st.write(f"**{selected} Tips:**")
    for tip in tips[selected]:
        st.write(f"- {tip}")

### 2. Update the Time Saver page with error handling ###

def page_time_predictor():
    st.header("⏱️ Cooking Time Predictor")
    
    ing_input = st.text_area(
        "Enter ingredients (comma separated)", 
        "chicken, rice, vegetables",
        key="time_predictor_input"  # Unique key
    )
    ing_count = st.slider(
        "Number of ingredients", 
        3, 20, 5,
        key="time_predictor_count"  # Unique key
    )
    
    if st.button("Predict", key="time_predictor_button"):
        try:
            model = load_time_prediction_model()
            pred_mins = model.predict([[ing_count, len(ing_input)]])[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Estimated Time", f"{pred_mins:.0f} minutes")
            with col2:
                st.metric("Complexity", 
                          "Easy" if pred_mins < 30 else 
                          "Medium" if pred_mins < 60 else "Hard")
            
            plot_cooking_tips(pred_mins)
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Try using simpler ingredients or reduce count")

### 3. Add to your PAGES dictionary ###


### 4. Add to your page router ###

# Machine Learning Models
@st.cache_resource
def build_models():
    # Load data
    df = load_data()
    
    # Calculate health score if it doesn't exist
    if 'health_score' not in df.columns:
        alpha = 0.003
        df['health_score'] = (df['protein_g'] * 2) - (df['fat_g'] + df['carb_g'] / 2) - (df['calories'] * alpha)
    
    # Recipe Recommender
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
    ing_matrix = vectorizer.fit_transform(df['ing_list'])
    svd = TruncatedSVD(n_components=50, random_state=42)
    reduced_features = svd.fit_transform(ing_matrix)
    nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
    nn_model.fit(reduced_features)
    
    # Nutrition Predictor
    df['ing_count'] = df['ing_list'].apply(len)
    df['has_meat'] = df['ingredients'].str.contains('chicken|beef|pork', case=False)
    features = ['ing_count', 'has_meat', 'cook_min']
    target = ['calories', 'protein_g', 'fat_g', 'carb_g']
    
    X = df[features].fillna(0)
    nutrition_models = {
        'calories': RandomForestRegressor().fit(X, df['calories'].fillna(0)),
        'protein_g': RandomForestRegressor().fit(X, df['protein_g'].fillna(0)),
        'fat_g': RandomForestRegressor().fit(X, df['fat_g'].fillna(0)),
        'carb_g': RandomForestRegressor().fit(X, df['carb_g'].fillna(0))
    }
    
    # Cooking Time Classifier
    bins = [0, 15, 30, 60, 120, np.inf]
    labels = ['Very Fast (<15m)', 'Fast (15-30m)', 'Medium (30-60m)', 'Slow (1-2h)', 'Very Slow (>2h)']
    df['time_category'] = pd.cut(df['cook_min'], bins=bins, labels=labels)
    
    # Filter out rows with missing data
    time_df = df.dropna(subset=['time_category', 'ing_list', 'ing_count'])
    
    time_vectorizer = TfidfVectorizer(max_features=100, tokenizer=lambda x: x, preprocessor=lambda x: x)
    X_ing = time_vectorizer.fit_transform(time_df['ing_list'])
    X_other = time_df[['ing_count']].fillna(0).values
    X_time = np.hstack([X_ing.toarray(), X_other])
    
    le = LabelEncoder()
    y_time = le.fit_transform(time_df['time_category'])
    time_model = GradientBoostingClassifier().fit(X_time, y_time)
    
    # Health Score Model - use only columns that exist
    health_cols = ['fat_g', 'carb_g', 'protein_g']
    health_df = df.dropna(subset=health_cols).copy()
    if 'health_score' in df.columns:
        health_df = health_df.dropna(subset=['health_score'])
    
    X_health = health_df[health_cols].fillna(0)
    y_health = health_df['health_score'] if 'health_score' in health_df.columns else (
        (health_df['protein_g'] * 2) - (health_df['fat_g'] + health_df['carb_g'] / 2) - (health_df['calories'] * alpha)
    )
    health_model = Ridge(alpha=1.0).fit(X_health, y_health)
    
    return {
        'recommender': (vectorizer, svd, nn_model),
        'nutrition': (nutrition_models, features),
        'time_classifier': (time_vectorizer, le, time_model),
        'health_model': health_model,
        'time_categories': labels
    }
models = build_models()

# ML Utility Functions
def recommend_similar_recipes(recipe_title, df):
    try:
        idx = df[df['title'] == recipe_title].index[0]
        vectorizer, svd, nn_model = models['recommender']
        recipe_features = svd.transform(vectorizer.transform([df.iloc[idx]['ing_list']]))
        distances, indices = nn_model.kneighbors(recipe_features)
        
        recommendations = []
        for i in range(1, len(indices[0])):
            rec_idx = indices[0][i]
            recommendations.append({
                'title': df.iloc[rec_idx]['title'],
                'category': df.iloc[rec_idx]['category'],
                'similarity': 1 - distances[0][i],
                'calories': df.iloc[rec_idx]['calories']
            })
        return pd.DataFrame(recommendations)
    except:
        return pd.DataFrame()

def predict_nutrition(ingredient_count, has_meat, cook_time):
    nutrition_models, feature_names = models['nutrition']
    input_data = pd.DataFrame([[ingredient_count, has_meat, cook_time]], 
                            columns=feature_names)
    
    predictions = {}
    for nutrient, model in nutrition_models.items():
        predictions[nutrient] = model.predict(input_data)[0]
    
    return predictions

def predict_cooking_time_category(ingredients):
    try:
        time_vectorizer, le, time_model = models['time_classifier']
        ing_list = [i.strip() for i in ingredients.split(';')]
        ing_features = time_vectorizer.transform([ing_list])
        other_features = np.array([[len(ing_list)]])
        features = np.hstack([ing_features.toarray(), other_features])
        
        pred = time_model.predict(features)
        return le.inverse_transform(pred)[0]
    except Exception as e:
        st.warning(f"Prediction failed: {str(e)}")
        return "Unknown"

def optimize_healthiness(current_fat, current_carb, current_protein):
    health_model = models['health_model']
    
    def objective(x):
        return -health_model.predict([[x[0], x[1], x[2]]])[0]
    
    cons = ({'type': 'ineq', 'fun': lambda x: x[2] - current_protein},
            {'type': 'ineq', 'fun': lambda x: current_fat - x[0]},
            {'type': 'ineq', 'fun': lambda x: current_carb - x[1]})
    
    x0 = [current_fat, current_carb, current_protein]
    bounds = [(0, None), (0, None), (0, None)]
    
    result = minimize(objective, x0, constraints=cons, bounds=bounds)
    
    return {
        'optimal_fat': result.x[0],
        'optimal_carb': result.x[1],
        'optimal_protein': result.x[2],
        'predicted_health_score': -result.fun
    }

# Data Processing
df = load_data()
df["health_score"] = (df["protein_g"] * 2) - (df["fat_g"] + df["carb_g"] / 2)
df["has_gluten"] = df["ingredients"].str.contains(r"(flour|wheat|barley)", flags=re.IGNORECASE, regex=True)
df["has_nuts"] = df["ingredients"].str.contains(r"(almond|cashew|walnut|pecan|hazelnut)", flags=re.IGNORECASE, regex=True)
df["has_dairy"] = df["ingredients"].str.contains(r"(milk|cheese|butter|cream)", flags=re.IGNORECASE, regex=True)
df['ing_count'] = df['ing_list'].apply(len)

# Pages
PAGES = {
    "Category Overview": "overview",
    "Nutrition Explorer": "nutrition",
    "Ingredients Lab": "ingredients",
    "Quick & Easy": "quick",
    "Publishing Trends": "trends",
    "Seasonality": "season",
    "Ingredient Pairs": "pairs",
    "Recipe Explorer": "recipe",
    "Meal Planner": "planner",
    "Health & Allergen Insights": "health",
    "For You": "personalized",
    "Time Saver": "timesaver",
}

with st.sidebar:
    st.title("Menu")
    page = st.radio("Go to", list(PAGES.keys()))
    st.markdown("---")
    st.write("Total recipes:", len(df))

def page_overview():
    st.header(" Average Calories per Category")
    cal_df = df.groupby("category").calories.mean().reset_index()
    st.altair_chart(
        alt.Chart(cal_df).mark_bar().encode(
            x="calories:Q", 
            y=alt.Y("category:N", sort="-x")
        ),
        use_container_width=True,
    )
    st.subheader("Compare Categories Side-by-Side")
    cats = st.multiselect("Select 2–3 categories", df['category'].unique(), default=["desserts", "salad"])
    if len(cats) >= 2:
        compare_df = df[df['category'].isin(cats)].groupby('category').agg({
            'calories': 'mean',
            'cook_min': 'median',
            'protein_g': 'mean'
        }).reset_index()
        st.dataframe(compare_df.style.background_gradient(), use_container_width=True)
    st.subheader("Top/Bottom Performers")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Healthiest (Low-Cal)")
        st.dataframe(df.nsmallest(5, 'calories')[['title', 'category', 'calories']])
    with col2:
        st.write("Most Indulgent")
        st.dataframe(df.nlargest(5, 'calories')[['title', 'category', 'calories']])
    st.markdown("---")
    if st.button("Surprise Me!"):
        facts = [
            f"Did you know? {len(df[df['cook_min'] < 15])} recipes take <15 min!",
            f"Desserts have {df[df['category']=='desserts']['calories'].mean():.0f} avg calories.",
            f"On average, recipes use {df.ing_list.apply(len).mean():.1f} ingredients!"
        ]
        st.success(random.choice(facts))

def page_nutrition():
    st.header("Nutrition vs Cooking Time")
    x_axis = st.selectbox("X‑axis", ["calories", "fat_g", "carb_g", "protein_g"])
    df_display = df[df["cook_min"].notnull()].copy()
    df_display["cook_hrs_bin"] = (df_display["cook_min"] // 60).astype(int)
    chart = alt.Chart(df_display).mark_circle(size=60, opacity=0.4).encode(
        x=alt.X(f"{x_axis}:Q", title=x_axis.capitalize()),
        y=alt.Y("cook_hrs_bin:O", title="Cooking Time (hrs, every hour)"),
        tooltip=["title", "calories", "fat_g", "carb_g", "protein_g", "cook_min", "cook_hrs_bin"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
    st.subheader("Filter by macros")
    max_cals = st.slider("Max calories", 0, int(df.calories.max()), 600)
    min_prot = st.slider("Min protein (g)", 0, int(df.protein_g.max()), 10)
    filt = df[(df.calories <= max_cals) & (df.protein_g >= min_prot)]
    st.write(f"{len(filt)} recipes match your criteria")
    st.dataframe(filt[["title", "calories", "protein_g", "cook_min"]])
    
def page_ingredients():
    st.header("Ingredients Lab")
    st.subheader("Most Common Ingredients")
    cat_choice = st.selectbox("Choose a category for common ingredients", 
                              options=["All"] + sorted(df.category.unique()))
    if cat_choice == "All":
        ing_series = pd.Series([i for sub in df.ing_list for i in sub]).str.lower()
    else:
        ing_series = pd.Series([i for sub in df[df.category == cat_choice].ing_list for i in sub]).str.lower()
    common = ing_series.value_counts().head(30).reset_index()
    common.columns = ["ingredient", "count"]
    st.altair_chart(
        alt.Chart(common).mark_bar().encode(
            x="count:Q",
            y=alt.Y("ingredient:N", sort="-x")
        ),
        use_container_width=True,
    )
    
    st.subheader("Ingredient Impact Analysis")
    selected_ing = st.selectbox("Analyze an ingredient", common['ingredient'].head(20))
    
    with_ing = df[df['ingredients'].str.contains(selected_ing, case=False)]
    without_ing = df[~df['ingredients'].str.contains(selected_ing, case=False)]
    
    if not with_ing.empty and not without_ing.empty:
        st.write(f"**Nutritional comparison (with vs without {selected_ing})**")
        comparison = pd.DataFrame({
            'With': with_ing[['calories', 'fat_g', 'carb_g', 'protein_g']].mean(),
            'Without': without_ing[['calories', 'fat_g', 'carb_g', 'protein_g']].mean()
        })
        st.dataframe(comparison.style.background_gradient())
    
def page_quick():
    st.header("Quick & Easy")
    quick = df[df.cook_min <= 30]
    st.write(f"{len(quick)} recipes ready in ≤30 min")
    st.dataframe(quick[["title", "cook_min", "calories", "category"]])
    st.download_button(
        "Export Quick Recipes",
        quick.to_csv(index=False).encode(),
        file_name="quick_recipes.csv",
        mime="text/csv",
    )
    st.subheader("Most complex recipes (ingredient count)")
    st.dataframe(df.nlargest(20, "ing_count")[["title", "ing_count", "category"]])
    
def page_trends():
    st.header("Publishing Trends (30‑day rolling)")
    ts = df.groupby("publish_date").size().rolling(30).mean().dropna().reset_index(name="count")
    st.line_chart(ts.set_index("publish_date"))
    
def page_season():
    st.header("Seasonal Calendar Heat‑map")
    df_cal = df.copy()
    df_cal["date"] = pd.to_datetime(df_cal["publish_date"])
    df_cal = df_cal.dropna(subset=["date"])
    df_cal["year"] = df_cal["date"].dt.year
    df_cal["month"] = df_cal["date"].dt.month
    df_cal["day"] = df_cal["date"].dt.day
    years = sorted(df_cal.year.unique())
    year_sel = st.selectbox("Select year", years, index=len(years) - 1)
    heat = (
        df_cal[df_cal.year == year_sel]
        .groupby(["month", "day"]).size().reset_index(name="count")
    )
    chart = alt.Chart(heat).mark_rect().encode(
        x=alt.X("day:O", title="Day"),
        y=alt.Y("month:O", title="Month"),
        color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), title="# Recipes"),
        tooltip=["month", "day", "count"]
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)
    
def page_pairs():
    st.header("Top Ingredient Co‑occurrences")
    top_n = st.slider("Show top N pairs", 10, 100, 20)
    @st.cache_data
    def compute_pairs():
        pair_counts = {}
        for lst in df.ing_list:
            unique_ings = set(i.lower() for i in lst)
            for a, b in itertools.combinations(sorted(unique_ings), 2):
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
        pc_df = (
            pd.DataFrame([(a, b, c) for (a, b), c in pair_counts.items()],
                         columns=["ing_a", "ing_b", "count"])
            .sort_values("count", ascending=False)
        )
        return pc_df
    pc_df = compute_pairs().head(top_n)
    st.dataframe(pc_df, use_container_width=True)
    
def page_recipe():
    st.header("Recipe Explorer")
    selected_cat = st.selectbox("Choose a category", options=["All"] + sorted(df.category.unique()))
    if selected_cat == "All":
        filtered_recipes = df.copy()
    else:
        filtered_recipes = df[df.category == selected_cat]
    if filtered_recipes.empty:
        st.error("No recipes available for the selected category!")
        return
    title = st.selectbox("Choose a recipe", sorted(filtered_recipes.title.unique()))
    row = filtered_recipes[filtered_recipes.title == title].iloc[0]
    st.subheader(row.title)
    st.write(f"Category: {row.category}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Calories", row.calories)
    c2.metric("Protein (g)", row.protein_g)
    c3.metric("Cooking time (min)", row.cook_min)
    st.markdown("### Ingredients")
    for ing in row.ing_list:
        st.write(f"• {ing}")
    
    st.markdown("### Similar Recipes")
    similar = recommend_similar_recipes(title, df)
    if not similar.empty:
        st.dataframe(similar)
    
    st.markdown("### Cooking Time Prediction")
    predicted_time = predict_cooking_time_category(";".join(row.ing_list))
    st.write(f"Predicted cooking time category: **{predicted_time}**")
    
    st.write("---")
    st.subheader("Similar Recipes")
    if st.button("Find Similar Recipes"):
        models = build_models(df)
        vectorizer, svd, nn_model = models['recommender']
        
        # Get similar recipes
        recipe_features = svd.transform(vectorizer.transform([row['ing_list']]))
        distances, indices = nn_model.kneighbors(recipe_features)
        
        similar_recipes = []
        for i in range(1, 5):  # Get top 4 similar recipes
            similar_idx = indices[0][i]
            similar_recipes.append(df.iloc[similar_idx])
        
        for recipe in similar_recipes:
            with st.expander(recipe['title']):
                st.write(f"Category: {recipe['category']}")
                st.write(f"Calories: {recipe['calories']}")
                st.write("Ingredients:")
                for ing in recipe['ing_list']:
                    st.write(f"- {ing}")
    
def page_planner():
    st.header("Weekly Meal Planner")
  
    col1, col2 = st.columns(2)
    with col1:
        min_cal = st.number_input("Min calories/day", 0, 12000, 1800, step=50)
        max_cal_default = max(min_cal + 200, 2200)
        max_cal = st.number_input("Max calories/day", min_cal, 12000, max_cal_default, step=50)
    with col2:
        prot_max_possible = int(df.protein_g.max(skipna=True))
        min_pro = st.number_input("Min protein g/day", 0, prot_max_possible, 90, step=5)
        max_pro_default = max(min_pro + 20, 150)
        max_pro = st.number_input("Max protein g/day", min_pro, prot_max_possible, max_pro_default, step=5)
    
    protein_cats_master = [
        "beef", "chicken", "turkey", "pork", "seafood",
        "pasta_and_noodles", "main_dishes"
    ]
    available_protein_cats = [c for c in protein_cats_master if c in df.category.unique()]
    if not available_protein_cats:
        st.error("No protein categories found in the dataset – cannot build Lunch slot.")
        return
    default_lunch = "chicken" if "chicken" in available_protein_cats else available_protein_cats[0]
    lunch_choice = st.selectbox("Choose lunch main category", available_protein_cats,
                                 index=available_protein_cats.index(default_lunch))
    
    cat_map = {
        "Breakfast": [c for c in ["breakfast_and_brunch", "breads"] if c in df.category.unique()],
        "Snack": [c for c in ["appetizers_snacks", "quick_meals_group", "side_dishes"] if c in df.category.unique()],
        "Salad": ["salad"] if "salad" in df.category.unique() else [],
        "Lunch": [lunch_choice],
        "Dinner": [c for c in ["main_dishes", "sheet_pan_dinners", "pasta_and_noodles"] if c in df.category.unique()],
        "Dessert": ["desserts"] if "desserts" in df.category.unique() else [],
    }
    
    for slot, cats in cat_map.items():
        if not cats:
            st.error(f"No categories available for slot '{slot}'. Adjust category mapping or dataset.")
            return
    
    base_pools = {}
    for slot, cats in cat_map.items():
        pool = df[df["category"].isin(cats)].dropna(subset=["calories", "protein_g"])
        pool = pool[np.isfinite(pool["calories"]) & np.isfinite(pool["protein_g"])]
        if pool.empty:
            st.error(f"Slot '{slot}' has no recipes after filtering. Try different categories or adjust dietary preference.")
            return
        base_pools[slot] = pool
    
    pools = {s: p.copy() for s, p in base_pools.items()}
    used_ids: set[int] = set()
    plan_rows = []
    
    for day in range(1, 8):
        prob = pulp.LpProblem(f"day_{day}", pulp.LpMinimize)
        sel = {}
        for slot, pool in pools.items():
            for idx in pool.index:
                sel[(slot, idx)] = pulp.LpVariable(f"sel_{slot}_{idx}", cat="Binary")
        for slot, pool in pools.items():
            prob += pulp.lpSum(sel[(slot, idx)] for idx in pool.index) == 1
        total_cal = pulp.lpSum(sel[(s, i)] * pools[s].loc[i, "calories"] for s in cat_map for i in pools[s].index)
        total_pro = pulp.lpSum(sel[(s, i)] * pools[s].loc[i, "protein_g"] for s in cat_map for i in pools[s].index)
        prob += total_cal >= min_cal
        prob += total_cal <= max_cal
        prob += total_pro >= min_pro
        prob += total_pro <= max_pro
        prob += 0
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        optimal = pulp.LpStatus[prob.status] == "Optimal"
        picks = {}
        for slot in cat_map:
            pool = pools[slot]
            if optimal:
                chosen_idx = next((idx for idx in pool.index if sel[(slot, idx)].value() == 1), None)
                chosen = pool.loc[chosen_idx] if chosen_idx is not None else None
            else:
                chosen = None
            if chosen is None:
                pool_unused = pool[~pool.index.isin(used_ids)]
                chosen = (pool_unused if not pool_unused.empty else pool).sample(1).iloc[0]
            picks[slot] = chosen
            used_ids.add(chosen.name)
            pools[slot] = pool.drop(index=chosen.name, errors="ignore")
            if pools[slot].empty:
                pools[slot] = base_pools[slot].copy()
        row = {
            "Day": f"Day {day}",
            **{slot: picks[slot].title for slot in cat_map},
            "Calories": int(sum(picks[s].calories for s in cat_map)),
            "Protein_g": int(sum(picks[s].protein_g for s in cat_map)),
        }
        plan_rows.append(row)
    
    plan_df = pd.DataFrame(plan_rows)
    st.dataframe(plan_df, use_container_width=True)
    st.download_button(
        "Download weekly plan (CSV)",
        plan_df.to_csv(index=False).encode(),
        file_name="weekly_meal_plan_full.csv",
        mime="text/csv",
    )

def page_health_allergen():
    st.header("Health & Allergen Insights")
    
    ALLERGEN_PATTERNS = {
        "gluten": [
        r"\b(flour|wheat|barley|pasta|rye|malt|bulgur|farina|spelt|african emmer|farro|seitan|triticale|bread|couscous|matzo|orge|udon|soba|dinkel|graham)\b"
    ],
    "nuts": [
        r"\b(almond|cashew|walnut|pecan|hazelnut|peanut|pistachio|macadamia|brazil nut|pine nut|nut butter|nut oil|pesto alla genovese|praline|marzipan|nougat|gianduja|nutella|nut meal)\b"
    ],
    "dairy": [
        r"\b(milk|cheese|butter|cream|yogurt|whey|casein|lactose|ghee|curds|custard|galactose|ice cream|kefir|lactalbumin|buttermilk|cream cheese|sour cream|whipped cream|half-and-half)\b"
    ]
    }

    def detect_allergens(ingredient_list: str) -> dict:
        low = str(ingredient_list).lower()
        return {
            allergen: any(re.search(pattern, low) for pattern in patterns)
            for allergen, patterns in ALLERGEN_PATTERNS.items()
        }

    # Apply allergen detection
    for allergen in ALLERGEN_PATTERNS:
        df[f"has_{allergen}"] = df["ingredients"].apply(
            lambda x: detect_allergens(x)[allergen]
        )

    alpha = 0.003
    
 
    def calculate_health_score(row):
        """Adjusted health score: protein benefit minus fat, carbs and a calorie penalty"""
        protein = row["protein_g"] if pd.notna(row["protein_g"]) else 0
        fat = row["fat_g"] if pd.notna(row["fat_g"]) else 0
        carbs = row["carb_g"] if pd.notna(row["carb_g"]) else 0
        calories = row["calories"] if pd.notna(row["calories"]) else 0
        return (protein * 2) - (fat + carbs / 2) - (calories * alpha)
    
    df["health_score"] = df.apply(calculate_health_score, axis=1)
    
    features = ["calories", "fat_g", "carb_g", "protein_g", "cook_min"]
    df_ml = df.dropna(subset=features + ["health_score"]).copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_ml[features], 
        df_ml["health_score"],
        test_size=0.2,
        random_state=42,
        stratify=pd.qcut(df_ml["health_score"], q=5)
    )
    
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("model", LinearRegression())
    ])
    
    pipeline.fit(X_train, y_train)
    
    st.subheader("Allergen-Free Recipes")
    
    required_allergen_cols = ["has_gluten", "has_nuts", "has_dairy"]
    if not all(col in df.columns for col in required_allergen_cols):
        st.error("Critical Error: Allergen detection failed!")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        gluten_free = st.checkbox("Gluten-Free", True)
    with col2:
        nut_free = st.checkbox("Nut-Free", True)
    with col3:
        dairy_free = st.checkbox("Dairy-Free", True)
    
    filtered = df.copy()
    if gluten_free:
        filtered = filtered[~filtered["has_gluten"]]
    if nut_free:
        filtered = filtered[~filtered["has_nuts"]]
    if dairy_free:
        filtered = filtered[~filtered["has_dairy"]]
    
    removed = df.drop(filtered.index)
    st.write(f"**Allergen-Free Recipes:** {len(filtered)}")
    
    if len(filtered) == 0:
        st.warning("No recipes match all selected allergen filters")
    else:
        st.dataframe(filtered[["title", "category"]].sort_values("title"), height=500)
    
    st.subheader("Healthiest Recipes")
    
    df_ml["prediction"] = pipeline.predict(df_ml[features])
    df_ml["confidence"] = 1 - (abs(df_ml["health_score"] - df_ml["prediction"]) / df_ml["health_score"].max())
    
    top_healthy = df_ml.nlargest(10, "prediction")
    
    for _, row in top_healthy.iterrows():
        with st.expander(f"{row['title']} (Score: {row['prediction']:.1f})"):
            st.write(f"**Category:** {row['category']}")
            st.write(f"**Calories:** {row['calories']} | **Protein:** {row['protein_g']}g")
            st.write(f"**Ingredients:** {row['ingredients'][:200]}...")
            
            if row['confidence'] > 0.95:
                st.success("Verified Health Rating")
            else:
                st.warning("Moderate Confidence Rating")
    
    

    def _run_validations():
        KNOWN_GLUTEN = ["bread", "pasta", "flour"]
        KNOWN_NUTS = ["almond", "walnut", "peanut"]
        KNOWN_DAIRY = ["cheese", "milk", "butter"]
        
        def _test_flag(ingredient, should_be_positive, allergen_type):
            result = detect_allergens(ingredient)[allergen_type]
            assert result == should_be_positive, \
                f"Allergen detection failed for {ingredient} (should be {should_be_positive} for {allergen_type})"
        
        for item in KNOWN_GLUTEN:
            _test_flag(item, True, "gluten")
        for item in KNOWN_NUTS:
            _test_flag(item, True, "nuts")
        for item in KNOWN_DAIRY:
            _test_flag(item, True, "dairy")
        
        _test_flag("water", False, "gluten")
        _test_flag("apple", False, "nuts")
        _test_flag("vinegar", False, "dairy")
        
        test_row = pd.Series({"protein_g": 20, "fat_g": 10, "carb_g": 40, "calories": 500})
        expected_score = (20 * 2) - (10 + 40/2) - (500 * alpha)
        assert abs(calculate_health_score(test_row) - expected_score) < 1e-6, "Health score formula error"
    
    try:
        _run_validations()
    except AssertionError as e:
        st.error(f"Validation Error: {str(e)}")
        st.stop()
def hybrid_recommendation_engine(df, fav_ingredients, health_goal):
    """Generate normalized recommendations (0-100% scale)"""
    if not fav_ingredients:
        fav_ingredients = list(keyword_bank().keys())  # Default to all categories
    
    # 1. Calculate ingredient match score (0-100)
    def calc_ingredient_match(recipe_ings):
        matched = sum(1 for ing in recipe_ings if ing in fav_ingredients)
        return min(100, (matched / max(1, len(fav_ingredients)) * 100) ) # Prevent division by zero
    
    df['ingredient_score'] = df['ing_list'].apply(calc_ingredient_match)
    
    # 2. Calculate health score (0-100)
    if health_goal == "High-Protein":
        health_score = (df['protein_g'] / df['protein_g'].max()) * 100
    elif health_goal == "Low-Carb":
        health_score = (1 - (df['carb_g'] / df['carb_g'].max())) * 100
    else:  # Balanced
        health_score = pd.Series(50, index=df.index)  # Neutral baseline as Series
    
    # Ensure health_score is a Series before clipping
    if isinstance(health_score, (int, float)):
        health_score = pd.Series(health_score, index=df.index)
    
    df['health_score'] = health_score.clip(0, 100)  # Now works with Pandas Series
    
    # 3. Combine scores (weighted average)
    df['match_score'] = (df['ingredient_score'] * 0.6) + (df['health_score'] * 0.4)
    
    return df.nlargest(5, 'match_score')[['title', 'category', 'ing_list', 'match_score', 'ingredient_score', 'health_score']] \
           .to_dict('records')

def page_personalized_recommendations():
    st.header("🍳 Recipes You'll Love")
    
    with st.form("preferences_form"):
        st.subheader("Tell us what you like")
        fav_ingredients = st.multiselect(
            "Favorite ingredients", 
            options=list(keyword_bank().keys()),
            key="fav_ingredients_multiselect"
        )
        health_goal = st.selectbox(
            "Health goal", 
            ["Balanced", "High-Protein", "Low-Carb"],
            key="health_goal_select"
        )
        submitted = st.form_submit_button("Get Recommendations")
    
    if submitted:
        try:
            recommended = hybrid_recommendation_engine(
                df.copy(),  # Work with a copy
                fav_ingredients, 
                health_goal
            )
            
            if not recommended:
                st.warning("No recipes match your preferences. Try different selections.")
                return
            
            st.success("Top Recommendations For You:")
            for i, recipe in enumerate(recommended, 1):
                with st.expander(f"#{i}: {recipe['title']} ({recipe['match_score']:.0f}% match)"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Category**: {recipe['category']}")
                        st.write(f"**Ingredients Score**: {recipe['ingredient_score']:.0f}%")
                    with col2:
                        st.write(f"**Health Score**: {recipe['health_score']:.0f}%")
                        st.write(f"**Goal**: {health_goal}")
                    
                    st.write("**Matching Ingredients**:")
                    matched_ings = [ing for ing in recipe['ing_list'] if ing in fav_ingredients]
                    st.write(", ".join(matched_ings[:5]) + ("..." if len(matched_ings) > 5 else ""))
                    
        except Exception as e:
            st.error(f"Recommendation failed: {str(e)}")
            st.info("Try selecting more ingredients or different health goals")
def train_time_model(df):
    df['ing_count'] = df['ing_list'].str.len()
    X = df[['ing_count', 'cook_min']].dropna()
    y = X.pop('cook_min')
    return RandomForestRegressor().fit(X, y)

def page_time_predictor():
    st.header("⏱️ Cooking Time Predictor")
    
    ing_input = st.text_area("Enter ingredients (comma separated)", "chicken, rice, vegetables")
    ing_count = st.slider("Number of ingredients", 3, 20, 5)
    
    if st.button("Predict"):
        # Load pre-trained model
        model = load_time_prediction_model()  
        pred_mins = model.predict([[ing_count, len(ing_input.split(","))]])[0]
        
        st.metric("Estimated Cooking Time", f"{pred_mins:.0f} minutes")
        plot_cooking_tips(pred_mins)



if page == "Category Overview":
    page_overview()
elif page == "Nutrition Explorer":
    page_nutrition()
elif page == "Ingredients Lab":
    page_ingredients()
elif page == "Quick & Easy":
    page_quick()
elif page == "Publishing Trends":
    page_trends()
elif page == "Seasonality":
    page_season()
elif page == "Ingredient Pairs":
    page_pairs()
elif page == "Recipe Explorer":
    page_recipe()
elif page == "Meal Planner":
    page_planner()
elif page == "Health & Allergen Insights":
    page_health_allergen()
elif page == "For You":
    page_personalized_recommendations()
elif page == "Time Saver":
    page_time_predictor()
