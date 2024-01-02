#Step 1: Import the necessary modules
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
%matplotlib inline

#Step 2: Reading the dataset
twd = pd.read_csv('twitter.csv')

#Step 3: Checking for null values
twd.isnull().sum()

#Step 4: Data Manipulation and Trnasformation
twd.rename(columns={'2401':'Tid','Borderlands':'entity','Positive':'sentiment','im getting on borderlands and i will murder you all ,':'tweet'},inplace=True)

#Renaming the categories
twd.loc[twd['sentiment']=='Irrelevant','sentiment'] = 'Neutral'

#Step 5: Droping duplicate values
twd = twd.drop_duplicates(['Tid'])

#Step 6: Feature Creation using the vader model
sia = SentimentIntensityAnalyzer()

reva = {}
for i, row in tqdm(twd.iterrows(),total=len(twd)):
    txt = row['tweet']
    ID = row['Tid']
    reva[ID] = sia.polarity_scores(txt)

vad = pd.DataFrame(reva).T
vad = vad.reset_index().rename(columns={'index':'Tid'})
twd = twd.merge(vad, how='right')

#Step 7: Data Visualization
#Entity vs positivity/neutrality/negativity
fig, axes = plt.subplots(3,1,figsize=(12,36))
sns.barplot(data=twd,y='entity',x='pos',order=twd.sort_values('pos').entity,ax=axes[0])
sns.barplot(data=twd,y='entity',x='neu',order=twd.sort_values('neu').entity,ax=axes[1])
sns.barplot(data=twd,y='entity',x='neg',order=twd.sort_values('neg').entity,ax=axes[2])
axes[0].set_title('Positive')
axes[1].set_title('Neutral')
axes[2].set_title('Negative')

#Sentiment vs positivity/neutrality/negativity
fig, axes = plt.subplots(3,1,figsize=(12,27))
sns.barplot(data=twd,y='sentiment',x='pos', ax=axes[0])
sns.barplot(data=twd,y='sentiment',x='neu', ax=axes[1])
sns.barplot(data=twd,y='sentiment',x='neg', ax=axes[2])
axes[0].set_title('Positive')
axes[1].set_title('Neutral')
axes[2].set_title('Negative')
