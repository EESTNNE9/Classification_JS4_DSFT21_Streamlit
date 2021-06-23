"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import pickle
import joblib,os
st.set_option('deprecation.showPyplotGlobalUse', False)

# Data dependencies


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
from wordcloud import WordCloud
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation


# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

tweet_vec  = joblib.load(open("resources/vectorizer.pkl", "rb"))

# Load your raw data
raw = pd.read_csv("resources/train.csv")
df = pd.read_csv("resources/CleanCleanTrain.csv")
df_clean = pd.read_csv("resources/Train_clean.csv")

# Map Sentiments 
df['sentiment'].replace(1, 'Positive', inplace=True)
df['sentiment'].replace(2, 'News', inplace=True)
df['sentiment'].replace(0, 'Neutral', inplace=True)
df['sentiment'].replace(-1, 'Negative', inplace=True)

# For word cloud
df_clean['sentiment'].replace(1, 'Positive', inplace=True)
df_clean['sentiment'].replace(2, 'News', inplace=True)
df_clean['sentiment'].replace(0, 'Neutral', inplace=True)
df_clean['sentiment'].replace(-1, 'Negative', inplace=True)

# Cleaner function 
def preprocess(post):
    from nltk.corpus import stopwords
    tknzr = TweetTokenizer()
    tokens = tknzr.tokenize(post.lower())
    stopwords = set(stopwords.words('english'))
    clean = [WordNetLemmatizer().lemmatize(token)for token in tokens if token.isalpha() and token not in punctuation and token not in stopwords]
    return ' '.join(clean) 



# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifier")
	st.subheader("Climate Change Tweet Classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["General Information","Data Exploration", "Predictions", "Functionality"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "General Information" page
	if selection == "General Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown('Climate change is arguably the most severe \
		and important challenge facing humanity during the 21st century. \
			Due to this, several companies might be considering focusing on offering products and services which \
				have minimum environmental impact or carbon footprint, in essence products which are environmentally friendly. \
					It would be valueable to these companies if they could determine how people perceive the threat of climate change \
						as this would help them gauge how their products and services may be perceived.')
		st.markdown("This Streamlit app aims to just to that by classifying text about climate change belief \
			into 4 distinct classes, **News**, **Positive**, **Negative** and **Neutral**. \
				The text classified as **Positive** could represent potential customers.")
		st.markdown("The models used were trained and developed using data scrapped from Twitter.")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']].head(20)) # will write the df to the page

	# Exploratory Data Analysis Page
	if selection == "Data Exploration":
		st.info("This is page is dedicated to visuals created to understand the problem statement and help guide the modeling process.")
		st.title("Data Exploration")

		# Sentiment Distribution
		st.markdown('Distribution of sentiments across the Tweets')
		fig = Figure()
		ax = fig.subplots()
		sns.countplot(x='sentiment', data=df, order=df['sentiment'].value_counts().sort_values(ascending=False).index, ax=ax)
		ax.set_xlabel('Sentiment')
		ax.set_ylabel('Count')
		ax.set_title('Number of Tweets per Sentiment')
		st.pyplot(fig)

		# Build a word cloud for positive Tweets
		st.markdown('Common Words in Positive Tweets')
		fig = Figure()
		ax = fig.subplots()
		mask = np.array(Image.open('twitter.jpg'))
		wordcloud = WordCloud(width = 1920, height=1080, background_color='white', mask=mask, max_font_size=100,max_words=100,collocations=False).generate(' '.join(df_clean['clean'][df_clean['sentiment'] == 'Positive']))
		plt.imshow(wordcloud, interpolation= 'bilinear')
		plt.axis('off')
		plt.show()
		st.pyplot()

		# Build a word cloud for tweets with negative sentiment tweets
		st.markdown('Common Words in Negative Tweets')
		wordcloud_neg = WordCloud(width = 1920, height=1080, background_color='white', mask=mask, max_font_size=100,max_words=100,collocations=False).generate(' '.join(df_clean['clean'][df_clean['sentiment'] == 'Negative']))
		plt.imshow(wordcloud_neg, interpolation='bilinear')
		plt.axis('off')
		plt.show()
		st.pyplot()

		# Build a word cloud for tweets with neutral sentiments
		st.markdown('Common Words in Neutral Tweets')
		wordcloud_neu = WordCloud(width = 1920, height=1080, background_color='white', mask=mask, max_font_size=100,max_words=100,collocations=False).generate(' '.join(df_clean['clean'][df_clean['sentiment'] == 'Neutral']))
		plt.imshow(wordcloud_neu, interpolation='bilinear')
		plt.axis('off')
		plt.show()
		st.pyplot()

		# Sentiment of different users 
		st.markdown('User sentiment, how different users feel about climate change.')
		plt.style.use("seaborn-colorblind")
		plt.figure(figsize=(10,6))
		sns.countplot(y='mention', hue='sentiment', data=df_clean, order=df_clean.mention.value_counts().iloc[:10].index)
		plt.xlabel('Number of mentions')
		plt.ylabel('User')
		plt.title('Top 10 users and their sentiment')
		plt.show()
		st.pyplot()


	
   
	# Building out the predications page
	if selection == "Predictions":
		st.info("Generate Predictions for your Tweet with our models.")

		data_source = ['Select Option', 'Text']
		
		source_selection = st.selectbox('What are you classifying?', data_source)

		# Getting the sentiment 
		def get_sentiment(val, my_dict):
			for key, value in my_dict.items():
				if val == value:
					return key

		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text (max. 280 characters):")
		input_text = preprocess(tweet_text)
		# models to use
		models = ['Select Option' ,'Logistic Regression', 'Naive Bayes', 'Random Forest']
		prediction_model = st.selectbox('Choose model', models)
		
		prediction_sentiments = {'Negative': -1,
		'Neutral':0, 
		'Positive':1,
		'News': 2}

		if st.button("Classify"):
			# Transforming user input with vectorizer
			
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice

			# Logistic Regression
			if prediction_model == 'Logistic Regression':
				vect_text = tweet_cv.transform([input_text]).toarray()
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
			
			# Naive Nayes
			if prediction_model == 'Naive Bayes':
				vect_text = tweet_vec.transform([input_text]).toarray()
				predictor = joblib.load(open(os.path.join("resources/lr_final_clean_unbalanced.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
			
			# Random Forest model, will keep it commented until we can find a proper model to use.
			if prediction_model == 'Random Forest':
				vect_text = tweet_vec.transform([input_text]).toarray() 
				predictor = joblib.load(open(os.path.join("resources/lgbm_class_clean_unbalanced.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
			
			
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			sentiment = get_sentiment(prediction, prediction_sentiments)
			st.success("Text Categorized as: {}".format(sentiment))
		
		# Functionality Page
	if selection == "Functionality":
		st.info("This page aims to give the user a **'behind the scenes'** view as to how the app works.\
			We will focus on two areas; how the data is **Preprocessed** and how predictions are generated using the respective **Models**.")
		st.subheader("Preprocessing")
		st.markdown('Human communication, both verbal and textual is quite advanced and unique, there is slang, jargon understood by specific people, abbreviations and the dreaded social media communication.\
			Unfortunately at the moment computers are limited in their capacity to understand the full depth of human language and communication...Hint hint NLP (Natural Language Processing).')
		st.markdown("NLP gives computers the ability to analyse, model and understand human language. Every intelligent application involving human language has some NLP behind it.\
			Some of the most most common applications of NLP we use in everday life include:\
				Modern search engines (**Google, Bing**),\
				Voice-based assistants (**Apple Siri, Amazon Alexa**),\
					Machine Translation services (**Google Translate**).")
		st.markdown("Word Tokenization")
		st.markdown("Lowercasing")
		st.markdown("Removal of punctuation, stopwords and any special characters")
		st.markdown("Lemmatization")
		st.markdown("Text Representation")
		#st.markdown()

		# Models Selection 
		st.header("Models")
		st.markdown('**Logistic Regression**: Logistic Regression allows us to predict the probability of an observation is of a certain class.\
			This model by default is a binary classifier, meaning the cannot handle target vectors with more than two classes.\
				To fit the problem statement we had, the model was extended to be a multi-class classifier using the OVR(one-vs-rest) extension.\
					With this, a separate model is trained for each class to predict if whether an observation is that class or not.')
		logistic_regression = Image.open("resources/Logistic Regression.png")
		st.image(logistic_regression, caption='Basic Logistic Regression Classifier (Source:https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc)')

		# Naive Bayes
		st.markdown("**Naive Bayes**: Naive Bayes has its foundations in the Bayes theorem, using Bayes' theorem it calculates the probability of observing a class label given the set\
			of features for the input data. This model assumes that each feature is independent of all other features and works well for text classification applications.")
		naive_bayes = Image.open("resources/bayes_new.png")
		st.image(naive_bayes, caption='Naive Bayes Classifier (Source:https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html)')

		# Random Forest
		st.markdown("**Random Forest**: Random Forest is an ensemble learning method whereby many decision trees are trained on a random sample of observations with replacement\
			that matches the original number of observations. This models makes predictions by voting to determine the predicted class.")
		random_forest = Image.open("resources/random.png")
		st.image(random_forest, caption="Random Forest Classifier (Source:https://jakevdp.github.io/PythonDataScienceHandbook/05.08-random-forests.html)")


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
