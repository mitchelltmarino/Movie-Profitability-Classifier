import os
import sys
import random
import numpy as np
import pandas as pd
import pprint
import json

# Dictionaries to store useful information.
DIRECTORS = dict()
STUDIOS = dict()
ACTORS = dict()
MOVIES = dict()
ASPECT_RATIOS = dict()

# Features to drop from IMDB dataset.
IMDB_DROPS = ['color', 'num_critic_for_reviews', 'director_facebook_likes',
              'actor_3_facebook_likes', 'actor_2_name', 'actor_1_facebook_likes',
              'actor_1_name', 'num_voted_users', 'cast_total_facebook_likes',
              'actor_3_name', 'facenumber_in_poster', 'movie_imdb_link',
              'num_user_for_reviews', 'actor_2_facebook_likes', 'imdb_score',
              'movie_facebook_likes']

# Features to drop from TMDB dataset.
TMDB_DROPS = ['budget', 'genres', 'homepage', 'keywords', 'original_language',
              'overview', 'popularity', 'production_countries', 'runtime', 'spoken_languages',
              'status', 'tagline', 'original_title', 'vote_average', 'vote_count']

# ---
# Import Movie Datasets.
# ---
# Import IMDB dataset.
imdb_movie_data = pd.read_csv(os.path.dirname(os.path.abspath(
    __file__))+"\\imdb-5000-movie-dataset\\movie_metadata.csv")
# Import TMDB dataset.
tmdb_movie_data = pd.read_csv(os.path.dirname(os.path.abspath(
    __file__))+"\\tmdb-5000-movie-dataset\\tmdb_5000_movies.csv")

# ---
# IMDB Processing.
# ---
# Adjust IMDB name.
for i, row in imdb_movie_data.iterrows():
    title = row['movie_title']
    try:
        imdb_movie_data.at[i, 'movie_title'] = title.strip()
    except Exception as e:
        pass

# ---
# TMDB Processing.
# ---
tmdb_movie_data.rename(columns={'title': 'movie_title'}, inplace=True)
tmdb_movie_data['title_year'] = tmdb_movie_data['release_date'].copy()

# Adjust Year
for i, row in tmdb_movie_data.iterrows():
    date = row['title_year']
    try:
        tmdb_movie_data.at[i, 'title_year'] = np.float64(date.split('-')[0])
    except Exception as e:
        pass

# Adjust TMDB name.
for i, row in tmdb_movie_data.iterrows():
    title = row['movie_title']
    try:
        tmdb_movie_data.at[i, 'movie_title'] = title.strip()
    except Exception as e:
        pass

# Change the release year to numeric.
tmdb_movie_data['title_year'] = pd.to_numeric(tmdb_movie_data['title_year'])

# ---
# Dropping Columns.
# ---
# Drop columns that we don't care about.
imdb_movie_data = imdb_movie_data.drop(columns=IMDB_DROPS)
tmdb_movie_data = tmdb_movie_data.drop(columns=TMDB_DROPS)

# ---
# Merging the Datasets.
# ---
# Inner join.
full_data = pd.merge(imdb_movie_data, tmdb_movie_data,
                     how='inner', on=['movie_title', 'title_year'])
# Drop duplicates.
full_data = full_data.drop_duplicates()
# Remove columns where country is not USA.
full_data = full_data.loc[full_data['country'] == 'USA']
# Release year no longer needed. Was only needed for join.
full_data = full_data.drop(columns=['title_year', 'country'])

# --
# Normalize Names Thus Far.
# --
full_data = full_data.rename(columns={'director_name': 'Director_Name', 'duration': 'Runtime', 'gross': 'Gross', 'genres': 'Genres', 'movie_title': 'Movie_Title',
                                      'plot_keywords': 'Plot_Keywords', 'language': 'Language', 'content_rating': 'Content_Rating', 'budget': 'Budget',
                                      'aspect_ratio': 'Aspect_Ratio', 'id': 'Movie_ID', 'production_companies': 'Production_Companies',
                                      'release_date': 'Release_Date', 'revenue': 'Revenue'})

# --
# Drop data that has no values for Gross (the target feature)
# --
full_data = full_data[np.isfinite(full_data['Revenue'])]

# --
# Import Credits Dataset.
# --
tmdb_credit_data = pd.read_csv(os.path.dirname(os.path.abspath(
    __file__))+"\\tmdb-5000-movie-dataset\\tmdb_5000_credits.csv")
full_data.to_csv(os.path.dirname(os.path.abspath(__file__))+"\\data_before_processing.csv", index=False)

# --
# Informative data processing..
# --
# Runtime
full_data['Budget'] = full_data['Budget'].replace(np.nan, np.float64(0))
full_data['Budget'] = full_data['Budget'].replace(np.float64(
    0), np.mean(pd.to_numeric(full_data['Budget'][full_data['Budget'] != np.float64(0)], errors='throw')))
print(np.mean(pd.to_numeric(full_data['Budget'][full_data['Budget'] != np.float64(0)], errors='throw')))

# --
# Accumulate movie info..
# --
# Get top three actors for each movie.
for i, row in full_data.iterrows():
    # Get the id of the movie.
    tmdb_movie_id = row['Movie_ID']
    # Store move information.
    if tmdb_movie_id not in MOVIES.keys():
        MOVIES[tmdb_movie_id] = dict()
        MOVIES[tmdb_movie_id]['title'] = row['Movie_Title']
        MOVIES[tmdb_movie_id]['gross'] = row['Gross']
        MOVIES[tmdb_movie_id]['revenue'] = row['Revenue']
        MOVIES[tmdb_movie_id]['budget'] = row['Budget']
        MOVIES[tmdb_movie_id]['lead_actor'] = None
        MOVIES[tmdb_movie_id]['director'] = None

# ---
# ACTOR DETAILS. --> Lead_Actor_ID, Lead_Actor_Name
# ---
# Create new columns for actors.
full_data['Lead_Actor_ID'] = np.float64(0)
full_data['Lead_Actor_Name'] = ""

# Get top three actors for each movie.
for i, row in full_data.iterrows():
    # Get the id of the movie.
    tmdb_movie_id = row['Movie_ID']
    # Find the credits for that movie.
    cast = tmdb_credit_data.loc[tmdb_credit_data['movie_id']
                                == tmdb_movie_id]['cast'].item()
    # Set Dataframe Info.
    try:
        cast_json = json.loads(cast)
        actor_id = cast_json[0]['id']
        full_data.at[i, 'Lead_Actor_ID'] = actor_id
        full_data.at[i, 'Lead_Actor_Name'] = cast_json[0]['name']
        # Store actor information.
        MOVIES[tmdb_movie_id]['lead_actor'] = cast_json[0]['name']
        if actor_id not in ACTORS.keys():
            ACTORS[actor_id] = dict()
            ACTORS[actor_id]['name'] = cast_json[0]['name']
            ACTORS[actor_id]['movies'] = set()
        ACTORS[actor_id]['movies'].add(tmdb_movie_id)
    except Exception as e:
        pass

# ---
# DIRECTOR DETAILS. --> Director_ID, Director_Name
# ---
# Create new columns for actors.
full_data['Director_ID'] = np.float64(0)
full_data['Director_Name'] = "Other"

# Get top three actors for each movie.
for i, row in full_data.iterrows():
    # Get the id of the movie.
    tmdb_movie_id = row['Movie_ID']
    # Find the credits for that movie.
    crew = tmdb_credit_data.loc[tmdb_credit_data['movie_id']
                                == tmdb_movie_id]['crew'].item()
    # Set Dataframe Info.
    try:
        crew_json = json.loads(crew)
        # Find director details in crew list.
        location = 0
        for j, crew_member in enumerate(crew_json):
            if crew_member['job'].lower() == 'director':
                location = j
                break
        # Reference director details.
        director_details = crew_json[location]
        director_id = director_details['id']
        full_data.at[i, 'Director_ID'] = np.float64(director_id)
        full_data.at[i, 'Director_Name'] = director_details['name']
        # Store actor information.
        MOVIES[tmdb_movie_id]['director'] = director_details['name']
        if director_id not in DIRECTORS.keys():
            DIRECTORS[director_id] = dict()
            DIRECTORS[director_id]['name'] = director_details['name']
            DIRECTORS[director_id]['movies'] = set()
        DIRECTORS[director_id]['movies'].add(tmdb_movie_id)
    except Exception as e:
        pass

# ---
# STUDIO DETAILS. --> Studio_IDs, Studio_Names
# Note: 4710 studios.
# ---
# Create new columns for actors.
full_data['Studio_IDs'] = np.empty((len(full_data), 0)).tolist()
full_data['Studio_Names'] = np.empty((len(full_data), 0)).tolist()

for i, row in full_data.iterrows():
    # Get the id of the movie.
    tmdb_movie_id = row['Movie_ID']
    studio_ids = []
    studio_names = []
    for studio in json.loads(row['Production_Companies']):
        studio_id = studio['id']
        studio_ids.append(studio_id)
        studio_names.append(studio['name'])
        # Store studio information.
        if studio_id not in STUDIOS.keys():
            STUDIOS[studio_id] = dict()
            STUDIOS[studio_id]['name'] = studio['name']
            STUDIOS[studio_id]['movies'] = set()
        STUDIOS[studio_id]['movies'].add(tmdb_movie_id)
    full_data.at[i, 'Studio_IDs'] = studio_ids
    full_data.at[i, 'Studio_Names'] = studio_names

# --
# Genres Dataset. One Hot Encoding.
# --
GENRES = set()
FULL_GENRES = dict()
genre_count=0

# Create a list of all existing genres in the data set.
for i, row in full_data.iterrows():
    genres = row['Genres'].split('|')
    for genre in genres:
        GENRES.add(genre)
    genre_list = row['Genres']
    if genre_list not in FULL_GENRES.keys():
        FULL_GENRES[genre_list] = genre_count
        genre_count += 1

# Create new columns for all genres, each of format: Genre_Comedy.
# The default value of each column will be 0.
for genre in GENRES:
    full_data['Genre_'+genre] = np.float64(0)
full_data['Full_Genre'] = np.float64(0)

# Assign 1 to the columns for each movie that has a given genre.
for i, row in full_data.iterrows():
    genres = row['Genres'].split('|')
    for genre in genres:
        full_data.at[i, "Genre_"+genre] = np.float64(1)
    full_data.at[i, 'Full_Genre'] = np.float(FULL_GENRES[row['Genres']])


# --
# - Release_Month
#    --> Ordinal (Label Encoding --> 1 to 12)
# --
# Create column for the release month.
full_data['Release_Month'] = np.float64(0)

for i, row in full_data.iterrows():
    release_date = row['Release_Date']
    # Release date in format YYYY-MM-DD
    release_month = np.float64(release_date.split('-')[1]) - 1
    # Set the release month for the movie.
    full_data.at[i, 'Release_Month'] = release_month

# --
# - Director_Avg_Movie_Revenue
#    --> Continuous, average director movie revenue.
# - Director_Movie_Count
#    --> Discrete.
# --

full_data['Director_Avg_Movie_Revenue'] = np.float64(0)
full_data['Director_Movie_Count'] = np.float64(0)
full_data['Director_Ratio'] = np.float(0)

for i, row in full_data.iterrows():

    try:
        # Get director details
        director_id = row['Director_ID']
        director_details = DIRECTORS[director_id]
        # Calculate total revenue.
        total_revenue = 0
        total_budget = 0
        for movie_id in director_details['movies']:
            total_revenue += MOVIES[movie_id]['revenue']
            total_budget += MOVIES[movie_id]['budget']
        # Number of movies director has done.
        movie_count = len(director_details['movies'])
        # Calculate average revenue.
        avg_revenue = round(total_revenue / movie_count, 2)
        # Ratio
        ratio = round(total_revenue / total_budget, 2)
        # Add director movie count and avg movie revenue to dataframe.
        full_data.at[i, 'Director_Avg_Movie_Revenue'] = np.float64(avg_revenue)
        full_data.at[i, 'Director_Movie_Count'] = np.float64(movie_count)
        full_data.at[i, 'Director_Ratio'] = np.float64(ratio)
    except Exception as e:
        pass

# --
# - Plot Keywords
#    --> Average of average revenue of the keywords.
# --

# Store all keywords and associated movies.
PLOT_KEYWORDS = dict()
for i, row in full_data.iterrows():
    try:
        # Note movie ID.
        movie_id = row['Movie_ID']
        # For each keyword in the movie's plot.
        for keyword in row['Plot_Keywords'].split('|'):
            # If keyword does not already exist.
            if keyword not in PLOT_KEYWORDS.keys():
                PLOT_KEYWORDS[keyword] = dict()
                PLOT_KEYWORDS[keyword]['movies'] = set()
            # Add the movie to the list of the keyword's movies.
            PLOT_KEYWORDS[keyword]['movies'].add(movie_id)
    except Exception as e:
        pass

# Calculate average revenue for all keywords.
for keyword, keyword_details in PLOT_KEYWORDS.items():

    # Calculate total revenue of all movies with the keyword.
    total_revenue = 0
    total_budget = 0
    for movie_id in keyword_details['movies']:
        total_revenue += MOVIES[movie_id]['revenue']
        total_budget += MOVIES[movie_id]['budget']
    # Calculate avg revenue.
    average_revenue = round(total_revenue / len(keyword_details['movies']), 2)
    PLOT_KEYWORDS[keyword]['Avg_Revenue'] = average_revenue
    # Ratio
    ratio = round(total_revenue / total_budget, 2)
    PLOT_KEYWORDS[keyword]['Ratio'] = ratio

# Update the dataframe with average revenue for keywords.
full_data['Keywords_Avg_Revenue'] = np.float64(0)
full_data['Keywords_Ratio'] = np.float64(0)

for i, row in full_data.iterrows():
    try:
        keywords = row['Plot_Keywords'].split('|')
        total_avg_revenue = 0
        total_avg_ratio = 0
        # Add up the averages of each keyword's revenue.
        for keyword in keywords:
            total_avg_revenue += PLOT_KEYWORDS[keyword]['Avg_Revenue']
            total_avg_ratio += PLOT_KEYWORDS[keyword]['Ratio']
        # Get the true average revenue for all the movies' keywords.
        avg_revenue = round(total_avg_revenue / len(keywords), 2)
        ratio = round(total_avg_ratio / len(keywords), 2)
        # Add director movie count and avg movie revenue to dataframe.
        full_data.at[i, 'Keywords_Avg_Revenue'] = np.float64(avg_revenue)
        full_data.at[i, 'Keywords_Ratio'] = np.float64(ratio)
    except Exception as e:
        pass

# --
#- Language
#    --> ONLY English
# Note: 4211 of the films are English.
# --

# --
# - Content_Rating_Score
#    --> Ordinal. (Label Encoding)
# --
full_data['Content_Rating_Score'] = np.float64(0)
# Content Ratings on a scale of maturity level.
rating_scale = {'G': 0,
                'TV-G': 0,
                'GP': 0,
                'Approved': 0,
                'Passed': 0,
                'PG': 1,
                'TV-PG': 1,
                'PG-13': 2,
                'TV-14': 2,
                'R': 3,
                'M': 3,
                'Unrated': 3,
                'Not Rated': 3,
                'NC-17': 4,
                'X': 4,
                'Disapproved': 4
                }
for i, row in full_data.iterrows():
    try:
        content_rating = row['Content_Rating']
        # If content rating is null, assume G.
        if not isinstance(content_rating, str) and np.isnan(content_rating):
            content_rating = 'G'
        # Set rating value based on scale.
        rating_score = rating_scale[content_rating]
        full_data.at[i, 'Content_Rating_Score'] = np.float64(rating_score)
    except Exception as e:
        pass

# --
# - Studio_Avg_Movie_Revenue
#    --> Continuous, average of all studio's average movie revenues.
# --

full_data['Studios_Avg_Movie_Revenue'] = np.float64(0)
full_data['Studios_Ratio'] = np.float64(0)

for i, row in full_data.iterrows():
    try:
        # Get director details
        studio_ids = row['Studio_IDs']
        # The summation of all studio averages.
        studios_total_avg_revenue = 0
        total_avg_ratio = 0
        for studio_id in studio_ids:
            studio_details = STUDIOS[studio_id]
            # Calculate total revenue.
            total_revenue = 0
            total_budget = 0
            for movie_id in studio_details['movies']:
                total_revenue += MOVIES[movie_id]['revenue']
                total_budget += MOVIES[movie_id]['budget']
            # Number of movies director has done.
            movie_count = len(studio_details['movies'])
            # Calculate average revenue.
            avg_revenue = total_revenue / movie_count
            # Add to the total revenue.
            studios_total_avg_revenue += avg_revenue
            # Ratio
            total_avg_ratio += total_revenue / total_budget
        # The studio's avg revenue
        studios_avg_revenue = round(
            studios_total_avg_revenue / len(studio_ids), 2)
        # The studio's avg ratio
        studios_avg_ratio = round(total_avg_ratio / len(studio_ids), 2)
        # Add director movie count and avg movie revenue to dataframe.
        full_data.at[i, 'Studios_Avg_Movie_Revenue'] = np.float64(
            studios_avg_revenue)
        full_data.at[i, 'Studios_Ratio'] = np.float64(
            studios_avg_ratio)
    except Exception as e:
        pass

# --
# - Lead_Actor_Avg_Movie_Revenue
#    --> Continuous, average actor income.
# - Lead_Actor_Movie_Count
#    --> Discrete.
# --
full_data['Lead_Actor_Avg_Movie_Revenue'] = np.float64(0)
full_data['Lead_Actor_Movie_Count'] = np.float64(0)
full_data['Lead_Actor_Ratio'] = np.float64(0)

for i, row in full_data.iterrows():
    try:
        # Get director details
        actor_id = row['Lead_Actor_ID']
        actor_details = ACTORS[actor_id]
        # Calculate total revenue.
        total_revenue = 0
        total_budget = 0
        for movie_id in actor_details['movies']:
            total_revenue += MOVIES[movie_id]['revenue']
            total_budget += MOVIES[movie_id]['budget']
        # Number of movies director has done.
        movie_count = len(actor_details['movies'])
        # Calculate average revenue.
        avg_revenue = round(total_revenue / movie_count, 2)
        # Ratio
        ratio = round(total_revenue / total_budget, 2)
        # Add director movie count and avg movie revenue to dataframe.
        full_data.at[i, 'Lead_Actor_Avg_Movie_Revenue'] = np.float64(
            avg_revenue)
        full_data.at[i, 'Lead_Actor_Movie_Count'] = np.float64(movie_count)
        full_data.at[i, 'Lead_Actor_Ratio'] = np.float64(ratio)
    except Exception as e:
        pass

DROP_FINAL = ['Gross', 'Production_Companies', 'Language']
full_data = full_data.drop(columns=DROP_FINAL)

# ---
# Replace missing Runtime with median 
# and
# Replace missing Aspect_Ratio median.
# ---
# Runtime
full_data['Runtime'] = full_data['Runtime'].replace(np.nan, np.float64(0))
full_data['Runtime'] = full_data['Runtime'].replace(np.float64(
    0), np.mean(pd.to_numeric(full_data['Runtime'][full_data['Runtime'] != np.float64(0)], errors='throw')))
print(np.mean(pd.to_numeric(full_data['Runtime'][full_data['Runtime'] != np.float64(0)], errors='throw')))

# Aspect_Ratio
full_data['Aspect_Ratio'] = full_data['Aspect_Ratio'].replace(np.nan, np.float64(0))
full_data['Aspect_Ratio'] = full_data['Aspect_Ratio'].replace(np.float64(
    0), np.median(pd.to_numeric(full_data['Aspect_Ratio'][full_data['Aspect_Ratio'] != np.float64(0)], errors='throw')))
print(np.median(pd.to_numeric(full_data['Aspect_Ratio'][full_data['Aspect_Ratio'] != np.float64(0)], errors='throw')))

# Class Creation.
full_data['Class'] = np.float64(0)

for i, row in full_data.iterrows():
    # Get revenue and budget.
    revenue = row['Revenue']
    budget = row['Budget']
    # Case 1. No Revenue.
    if revenue == 0:
        # Failue.
        full_data.at[i, 'Class'] = np.float64(0)
    # Case 2. Some revenue, but less than budget.
    elif revenue > 0 and revenue < budget:
        # Extreme Success.
        full_data.at[i, 'Class'] = np.float64(0) 
    # Case 3. Revenue is greater than budget.
    elif revenue < 2*budget:
        # Profitable.
        full_data.at[i, 'Class'] = np.float64(1) 
    # Case 3. Revenue is double that of budget.
    else:
        # Breakeven.
        full_data.at[i, 'Class'] = np.float64(1)

# Clamp Normalization.
clamp_list = ['Budget', 'Revenue', 'Runtime', 'Aspect_Ratio', 'Director_Avg_Movie_Revenue',
       'Director_Movie_Count', 'Keywords_Avg_Revenue',
       'Studios_Avg_Movie_Revenue', 'Lead_Actor_Avg_Movie_Revenue',
       'Lead_Actor_Movie_Count', 'Director_Ratio', 'Keywords_Ratio',
       'Studios_Ratio', 'Lead_Actor_Ratio']
for column in clamp_list:
    # IQR
    Q1 = full_data[column].quantile(0.25)
    Q3 = full_data[column].quantile(0.75)
    IQR = Q3 - Q1
    # Lower and Upper
    lower_threshold = Q1 - (1.5 * IQR)
    upper_threshold = Q3 + (1.5 * IQR)
    # Full Data Replacements.
    full_data.loc[full_data[column] < lower_threshold, column] = lower_threshold
    full_data.loc[full_data[column] > upper_threshold, column] = upper_threshold

# Output final merged & massaged dataset ready for machine learning.
full_data.to_csv(os.path.dirname(os.path.abspath(__file__))+"\\processed_data_final.csv", index=False)
