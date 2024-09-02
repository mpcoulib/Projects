#!/usr/bin/env python
# coding: utf-8

# Installation of packages, libraries

# In[ ]:


get_ipython().system('pip install python-dotenv')
get_ipython().system('pip install requests')
get_ipython().system('pip install pybase64')
get_ipython().system('pip install billboard.py')


# Importing necessary libraries

# In[ ]:


# from dotenv import load_dotenv
import pybase64
from requests import*
import os
import json
import billboard
import pandas as pd


# First of all let's get the most popular songs from 2012 to 2023 by using the billboard API. We will retrieve the songs that made it to the Global top 200 and consider them as popular.
# Let's use the 2020 as an example

# In[ ]:


chart = billboard.ChartData("billboard-global-200", date=None, year=2020, fetch=True, timeout=25)


# # API Request for popular and Non-popular songs

# In here we get the token and format it to make the API requests from Spotify
# 
# 

# In[ ]:


# load_dotenv()
client_id = 'b4e9cab6fd9d4f3fa4beeb221fbb79ac'
client_secret = 'ce526a94dc614f9c84759d78108d3797'


# client_id = 'f01cc581db6649d692e31874288d4d17'
# client_secret = 'd48e9ce7b46347e4bb2083d2395f03b4'
print(client_id, client_secret)
def get_token():
    auth_string = f"{client_id}:{client_secret}"
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(pybase64.b64encode(auth_bytes), "utf-8")

    url= "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    result = post(url, headers=headers, data=data)
    json_result= json.loads(result.content)
    token = json_result['access_token']
    return token

token = get_token()
print(token)
def get_auth_header(token):
    return {"Authorization": "Bearer " + token}


# In[ ]:


# tiktok = billboard.ChartData("tikTok-billboard-top-50", weeks='1')
# tiktok


# We move to building the search track function that will take in a token and the title of the song to search for it on spotify.
# 
# Here we are trying to mainly get the id of the song.
# We are doing that to be able to use that id when searching for specific audio features

# In[ ]:


def search_for_track(token, name):
    url= "https://api.spotify.com/v1/search"
    headers = get_auth_header(token)
    query= f"?q={name}&type=track&limit=1"
    query_url= url + query
    result= get(query_url, headers=headers)
    print(result.content)
    if len(json.loads(result.content)["tracks"]["items"])==0:
      return "No track"
    else:
      song_name = json.loads(result.content)["tracks"]["items"][0]["name"]
      artist_name = json.loads(result.content)["tracks"]["items"][0]["artists"][0]["name"]
      song_id= json.loads(result.content)["tracks"]["items"][0]["id"]
    return song_name, artist_name, song_id
# print(search_for_track(token, 'vih'))


# We are generating random characters to search for random songs on spotify and make sure they are not apart of the popular song list before adding them to the list

# In[ ]:


import random
import string

def generate():
  words=[]
# Set the required length
  length = 5
  for i in range(2000):
    characters = string.ascii_letters
    generated_string = ''.join(random.choice(characters) for _ in range(length))
    words.append(generated_string)
  return words
print(generate())



# Let us create lists to store the song titles, artists names, years, and popularity
# 
# Then let's request for all the songs that made it to the top global 200 for the past 24 years. We then add them to the lists with their popularity set to 1.

# In[ ]:


titles=[]
artists=[]
popularity=[]
chart_list=[]

for y in range(2006, 2023):
  # chart = billboard.ChartData("pop-songs", year=y)
  chart = billboard.ChartData("Hot-100-Songs", date=None, year=y, fetch=True, timeout=25)
  for song in chart:
    # print(song)
    s=song.title + " "+ song.artist
    if s not in chart_list:
      chart_list.append(s)
      # print(chart_list)
      titles.append(song.title)
      artists.append(song.artist)
      popularity.append(1)


# Let's use the generated random characters to search for specific songs and add them to the list of titles and artists and set the popularity column to 1.

# In[ ]:


title2=[]
artist2=[]
popularity2=[]
song_id=[]
for w in generate():
  track_details=search_for_track(token, w)
  if track_details != "No track":
    if track_details[2] not in song_id:
      title2.append(track_details[0])
      artist2.append(track_details[1])
      popularity2.append(0)
      song_id.append(track_details[2])



# print(titles)


# In[ ]:


popular = pd.DataFrame({"Title":titles,
                        "Artist":artists,
                        "Popular": popularity} )


popular


# In[ ]:


unpopular = pd.DataFrame({"Title":title2,
                        "Artist":artist2,
                          "track_id":song_id,
                        "Popular": popularity2} )


# In[ ]:


unpopular


# Now that we have this dataFrame, We need to have the songs features from spotify

# First retrival technique: Straight from the official spotify API https://developer.spotify.com/documentation/web-api

# Let us create the feature columns now

# In[ ]:


track=search_for_track(token, 'As It Was')[2]
print(track)
url= f"https://api.spotify.com/v1/audio-features/{track}"
headers = get_auth_header(token)
result= get(url, headers=headers)
json_result= json.loads(result.content)
feature=[]
for i in json_result:
  feature.append(i)
print(feature)
for f in feature:
  popular[f]=''
  unpopular[f]=''
unpopular


# In[ ]:


unpopular['track_id']


# Now that we have a function that is going to search for the track in the general search engine of spotify, we are going to proceed in two steps to retrieve the songs features from the spotify API.
# 
# First, we will extract the song id from the result we are getting here and add it to the Popular pandas dataFrame.
# 
# Second, we use these ids to pull the tracks details and features.
# These two can be done within the same function it is just going to be a long one and will have a lot of for loops

# In[ ]:


def get_song():
  headers = get_auth_header(token)
  popular['track_id']=''
  for ind in popular.index:
    track_name=  popular['Title'][ind]
    tr= search_for_track(token, track_name)[2]
    if tr=='No track':
      popular['Popular']=0
    else:
      url= f"https://api.spotify.com/v1/audio-features/{tr}"
      result= get(url, headers=headers)
      json_result= json.loads(result.content)
      popular['track_id'][ind]=tr
      for i in json_result:
        popular[i][ind]= json_result[i]
get_song()


# In[ ]:


popular


# In[ ]:


def get_unpopular_song():
  headers = get_auth_header(token)
  for ind in unpopular.index:
    tr =  unpopular['track_id'][ind]
    url= f"https://api.spotify.com/v1/audio-features/{tr}"
    result= get(url, headers=headers)
    json_result= json.loads(result.content)
    print(json_result)
    if json_result !={'error': {'status': 429}}:
      for i in json_result:
      # print(i)
        unpopular[i][ind]= json_result[i]
get_unpopular_song()


# In[ ]:


unpopular


# In[ ]:


popular


# In[ ]:





# Now that we have the popular data from the billboard (Songs that made it to the Billboard).
# 
# The next step is to collect data that is regular( Did not make it in the billboard). For this, we will randomly generate data from spotify using the search method.

# In[ ]:


def get_unpopular(words):
  for i in words:
    tr=search_for_track(token, track_name)
    title= tr[0]
    artist= tr[1]
    song_id= tr[2]
    popular["Title"]=title
    popular['Artist']=artist
    popular['id']=song_id
    url= f"https://api.spotify.com/v1/audio-features/{song_id}"
    result= get(url, headers=headers)
    json_result= json.loads(result.content)
    for i in json_result:
      popular[i]= json_result[i]
    popular['track_id'][ind]=tr

  return


# In[ ]:


from googleapiclient.discovery import build
import pandas as pd
import seaborn as sns
import csv
import os


# In[ ]:


#api1 = AIzaSyCJKMoGl5qYP_VcvE_WmQzjqgjtBqJqFyY
#api2 = AIzaSyCiAKyNXn1WMvNHRrK1QTzvGJQzOpUNz4Y
#api3 = AIzaSyBW7wR6w3Fd0wcWWgH8ZvoYCFhEzOFkqlY
# --------------------------------------------------
# api4=AIzaSyDPYhiKGgnkGLQCUD_Unli54zZvLZMwmjQ
# api5=AIzaSyAG28AT0iDyHWvCGZl5I3NilCkifDC5mcE
# api6=AIzaSyCjmMVW3EE4gGVmyyPbcUVUL74Ke3HdJVc
# api7=AIzaSyAkLLb_Xt0zldbXF1SXgOLhAwo3XGP2C6o
# api8=AIzaSyC71ca2GIHjRaIvFSlBEJ_aTmsVexjYU0I
# api9=AIzaSyB0dJgnt7taTHuqLX0oQoLpFAYdnoN2Xwg
api_key = "AIzaSyB0dJgnt7taTHuqLX0oQoLpFAYdnoN2Xwg"
channel_id = "UCmGeQhltcV0XvFQQZPyJRGg"

youtube = build('youtube', 'v3', developerKey = api_key)


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


l = []
def search_by_keyword(query):
    # Define search parameters
    # search_query = query
    max_results = 2

    # Perform the search
    search_response = youtube.search().list(
        q=query,
        type='video',
        part='id,snippet',
        maxResults=max_results
    ).execute()

    # Iterate through the search results and print video titles
    #video_id = search_response.get('items', [])[0]['id']['videoId']
    try:
      video_id = search_response.get('items', [])[0]['id']['videoId']
    except:
      video_id = search_response.get('items', [])[0]['id']['videoId']
    title = search_response.get('items', [])[0]['snippet']['title']
    return video_id
    # for item in search_response.get('items', []):
    #     video_id = item['id']['videoId']
    #     title = item['snippet']['title']
    #     l.append(video_id)
    #     print(f'[{video_id}] Title: {title}')

# Call the function to perform the search
#print(search_by_keyword("Harry Styles, As It Was"))


# In[ ]:


import os
import googleapiclient.discovery


# In[ ]:



# Set your API key and API version
api_key = 'AIzaSyBxBd6Z2jeZ3dttrxQs1x-mqIVVHy6cOT4'
api_version = 'v3'

import os
import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Set your API key, client_id, and client_secret
client_id = '96168956525-baochdqrpe1kgcie5k2qhh1ksl4sir7c.apps.googleusercontent.com'
client_secret = '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'

# Set the video ID for the video you want to check
video_id = 'H5v3kku4y6Q'

# Set the specific date for which you want to retrieve the view count (in the format YYYY-MM-DD)
specific_date = '2023-01-01'

# Authenticate using OAuth 2.0
try:
    creds, _ = google.auth.default()
except google.auth.exceptions.DefaultCredentialsError:
    # If no credentials are available, request them
    flow = google.auth.OAuth2WebServerFlow(client_id, client_secret, 'https://www.googleapis.com/auth/yt-analytics.readonly')
    creds = google.auth.transport.requests.Request()
    creds = flow.step2_exchange()

# Create a YouTube Analytics API service
youtube_analytics = build('youtubeAnalytics', 'v2', credentials=creds)

# Get the view count for the specific date
try:
    response = youtube_analytics.reports().query(
        ids='channel==MINE',
        metrics='views',
        dimensions='video',
        filters=f'video=={video_id};startDate={specific_date};endDate={specific_date}',
    ).execute()

    for row in response['rows']:
        print(f'Views on {specific_date}: {row[1]}')
except HttpError as e:
    print(f'An error occurred: {e}')


# In[ ]:


def videos_by_id(video_id):
  max_results = 2
  videos_response = youtube.videos().list(
      part='id,statistics',
      id = video_id,
      maxResults=max_results
      ).execute()
  return videos_response


# In[ ]:


print(videos_by_id("H5v3kku4y6Q"))


# In[ ]:


popular = pd.read_csv('/content/drive/My Drive/Capstone/unpopular_song.csv')
popular


# In[ ]:


# popular['video_id'] = ""
# popular['youtube_view_count'] = ""
# popular['youtube_like_count'] = ""
# popular['youtube_comment_count'] = ""


# def get_video():
#   for ind in popular.index:
#     query_name= popular['Artist'][ind] + " " + popular['Title'][ind]
#     print(query_name)
#     item = search_by_keyword(query_name)
#     popular['video_id'][ind]=item
#     video_data=videos_by_id(item)
#     print(video_data)
#     #Store the youtube video data in the pandas dataframe
#     popular['youtube_view_count'][ind]= video_data["items"][0]["statistics"]["viewCount"]
#     popular['youtube_like_count'][ind]= video_data["items"][0]["statistics"]["likeCount"]
#     popular['youtube_comment_count'][ind]= video_data["items"][0]["statistics"]["commentCount"]

# get_video()


# In[ ]:


popular['video_id'] = ""
popular['youtube_view_count'] = ""
popular['youtube_like_count'] = ""
popular['youtube_comment_count'] = ""

def get_video():
    for ind in popular.index:
        # Check if the index is greater than or equal to 52
        if ind >= 1788:
            query_name = popular['Artist'][ind] + " " + popular['Title'][ind]
            print(query_name)
            item = search_by_keyword(query_name)
            popular['video_id'][ind] = item
            video_data = videos_by_id(item)
            print(video_data)
            # Store the youtube video data in the pandas dataframe
            #popular['youtube_view_count'][ind] = video_data["items"][0]["statistics"]["viewCount"]
            try:
              popular['youtube_view_count'][ind] = video_data["items"][0]["statistics"]["viewCount"]
            except:
              popular['youtube_view_count'][ind] = video_data["items"][0]["statistics"]["viewCount"]
            #popular['youtube_like_count'][ind] = video_data["items"][0]["statistics"]["likeCount"]
            try:
              popular['youtube_like_count'][ind] = video_data["items"][0]["statistics"]["likeCount"]
            except:
              popular['youtube_like_count'][ind] = video_data.get('likeCount', 0)
            #popular['youtube_comment_count'][ind] = video_data["items"][0]["statistics"]["commentCount"]
            try:
              popular['youtube_comment_count'][ind] = video_data["items"][0]["statistics"]["commentCount"]
            except:
              popular['youtube_comment_count'][ind] = video_data.get('commentCount', 0)

get_video()


# In[ ]:


popular[1788:]


# In[ ]:


def retrieve_and_save_data():

    today = datetime.date.today()
    log_path = '/content/drive/My Drive/Capstone/log.csv'

    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as log_file:
            csv.writer(log_file).writerow(['Date', 'Quota_Usage'])

    with open(log_path, 'r') as log_file:
        log_data = list(csv.DictReader(log_file))

    if not any(entry['Date'] == str(today) for entry in log_data):
        # Request and process data from YouTube API


        # Update quota usage in log
        with open(log_path, 'a', newline='') as log_file:
            csv.writer(log_file).writerow([str(today), '10,000'])

# Call the function to retrieve and save data
retrieve_and_save_data()


# In[ ]:


popular [449:155]


# In[ ]:


file_path = '/content/drive/My Drive/Capstone/unpopular25.csv'
df = popular[1788:]
# Save the DataFrame to CSV
df.to_csv(file_path, index=False)


# In[ ]:


from numpy.lib.function_base import append
final_popular=pd.read_csv(f'/content/drive/My Drive/Capstone/popular1.csv')
final_popular


# In[ ]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


import glob
import re

combined = pd.read_csv('/content/gdrive/MyDrive/Capstone/popular1.csv')
for i in range(2,19):
  file_path = '/content/gdrive/MyDrive/Capstone/popular' + str(i) +'.csv'
  df_new = pd.read_csv(file_path)
  combined = pd.concat([combined, df_new],ignore_index=True)
final_file = '/content/gdrive/MyDrive/Capstone/popular_with_yt.csv'
df1 = combined
# Save the DataFrame to CSV
# df1.to_csv(final_file, index=False)


# In[ ]:


combined = pd.read_csv('/content/gdrive/MyDrive/Capstone/unpopular1.csv')
for i in range(2,26):
  file_path = '/content/gdrive/MyDrive/Capstone/unpopular' + str(i) +'.csv'
  df_new = pd.read_csv(file_path)
  combined = pd.concat([combined, df_new],ignore_index=True)
final_file = '/content/gdrive/MyDrive/Capstone/unpopular_with_yt.csv'
df2 = combined
# Save the DataFrame to CSV
# df2.to_csv(final_file, index=False)


# In[ ]:


df1


# In[ ]:


df2


# # Initial Data analysis, visualization and feature engineering

# In this section of our project, we will be using the data to try to draw some correlations

# In[ ]:


def clean_unpopular(a, b):
  pid=list(a['id'])
  uid=list(b['id'])
  for ind in b.index:
    if b['id'][ind] in pid:
      b.drop(index=ind, inplace=True)
  return 'success'
clean_unpopular(df1, df2)


# In[ ]:


df2


# In[ ]:


final_path = '/content/gdrive/MyDrive/Capstone/unpopular_with_yt.csv'
df2.to_csv(final_path, index=False)


# Final cleaned data

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


popular=pd.read_csv('/content/gdrive/MyDrive/Capstone/popular_with_yt.csv')
popular['Popular']=1
popular


# In[ ]:


unpopular=pd.read_csv('/content/gdrive/MyDrive/Capstone/unpopular_with_yt.csv')
unpopular['Popular']=0
unpopular


# In[ ]:


df=pd.concat([unpopular, popular])
df=df[['Title', 'Artist', 'track_id', 'Popular', 'danceability', 'energy',
       'key', 'loudness', 'mode', 'speechiness', 'acousticness',
       'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature', 'youtube_view_count', 'youtube_like_count',
       'youtube_comment_count']]
df


# In[ ]:


df.isnull().sum()


# In[ ]:


df=df.dropna()


# In[ ]:


dup=df[df.duplicated(subset='track_id', keep=False )]
dup


# In[ ]:


file_path = '/content/gdrive/MyDrive/Capstone/dup.csv'
# Save the DataFrame to CSV
dup.to_csv(file_path, index=False)
dup_clean=pd.read_csv('/content/gdrive/MyDrive/Capstone/dup_clean.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


dup_clean


# In[ ]:


df=df.drop_duplicates(subset='track_id', keep=False)
df


# In[ ]:


df=pd.concat([df, dup_clean])
df.reset_index(inplace=True)
df


# In[ ]:


df


# In[ ]:


file_path = '/content/gdrive/MyDrive/Capstone/final_dataset.csv'
# Save the DataFrame to CSV
df.to_csv(file_path, index=False)


# Initial model

# In[ ]:




