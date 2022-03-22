"""**Code for Web Application**"""

import streamlit as st
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np

from wordcloud import WordCloud, STOPWORDS
import pickle

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from textblob import TextBlob

import streamlit.components.v1 as components
from streamlit import components

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

import gensim
from gensim import corpora

from PIL import Image

import json

import pyLDAvis
import pyLDAvis.gensim_models

seed = 4353


def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.sidebar.title("About")
    st.sidebar.info("Analysis of European Hotel Reviews Dataset")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ["Homepage", "Power BI Visualisation", "Sentiment Analysis", "Topic Modelling"])

    if page == "Homepage":
        st.title("Objectives of the project")
        st.write("The purpose of this project is to perform an analysis on a European Luxury Hotel Reviews dataset.")
        st.write("Deliverables of the project: -")
        st.write("1. An interactive Power BI Dashboard to visualize useful findings from the dataset. \n"
                 "2. Sentiment Analysis on the reviews data, including a Sentiment Classification Model using a suitable Machine Learning (ML) Algorithm. \n"
                 "3. Topic Modelling to discover the most talked topics in the reviews data.\n")
        
        st.header("Data Pipeline")
        image = Image.open("/content/drive/MyDrive/Colab Notebooks/Hotel_Analysis/Images/Data Pipeline.png")
        st.image(image, caption='Data Pipeline')
        
        st.header("Dataset Overview")
        st.write("""
        Original Dataset obtained from:
        https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe
                 """)
        data = load_data_ori()
        st.write(data.head(10))

        st.header("Data Shape")
        nRow, nCol = data.shape
        st.write(f'There are {nRow} rows and {nCol} columns in the dataset.')

        st.header("Metadata (Description of Data)")
        with st.expander("View more"):
          st.subheader("Dataset Variables")
          st.write("1. **Hotel_Address** - Address of the hotel")
          st.write("2. **Review_Date** -The date which the reviewer posted the review")
          st.write("3. **Average_Score** - The hotel's average score derived using the most recent review from the previous year.")
          st.write("4. **Hotel_Name** -	Name of the hotel")
          st.write("5. **Reviewer_Nationality**	- Nationality of the reviewer")
          st.write("6. **Negative_Review** - Negative review the reviewer gave to the hotel. 'No Negative' is indicated if the reviewer does not provide a negative review. Some basic text pre-processing (punctuation and Unicode removal) has been performed on the reviews.")
          st.write("7. **Review_Total_Negative_Word_Counts**	- Total number of words in the negative review")
          st.write("8. **Positive_Review** - Positive review the reviewer gave to the hotel. 'No Positive' is indicated if the reviewer does not provide a positive review. Some basic text pre-processing (punctuation and Unicode removal) has been performed on the reviews.")
          st.write("9. **Review_Total_Positive_Word_Counts** -	Total number of words in the positive review")
          st.write("10. **Reviewer_Score**	- Score that the reviewer has awarded the hotel")
          st.write("11. **Total_Number_of_Reviews_Reviewer_Has_Given** -	The number of previous reviews that the reviewers have given")
          st.write("12. **Total_Number_of_Reviews**	- The hotel's total amount of reviews")
          st.write("13. **Tags** -	Additional information regarding the reviewer / what the review given was based on. (E.g., travel type, number of guests, hotel room type, days of stay etc.)")
          st.write("14. **days_since_review**	 - The time between the date of the review and the date of the scrape.")
          st.write("15. **Additional_Number_of_Scoring** -	Some guests only left a score for the hotel rather than writing a review. This attribute reflects the number of valid scores without a textual review.")
          st.write("16. **lat** -	Latitude of the hotel)")
          st.write("17. **lng** -	Longitude of the hotel")


    elif page == "Power BI Visualisation":
        st.title("Power BI Dashboard 📊")
        with st.expander("See details"):
          st.write("**Page 1: Overview of European Hotels** - Have a closer look at where each hotel is located at, scores that the hotels received from their reviewers, as well as get a better idea of each hotel's popular visitor types etc.")
          st.write("**Page 2: Hotel Reviewer Analysis** - Have a closer look at the best and worst hotels in Europe.")
          st.write("**Page 3: Sentiment Analysis** - Have a closer look at the most mentioned negative and positive words in the reviews and the sentiment counts of each hotel.")
        st.markdown('<iframe title="Hotel (Web version)" width="800" height="486" src="https://app.powerbi.com/view?r=eyJrIjoiZWViNWE4N2YtZmM0OS00MzFkLTk0MDgtMDdlOTZhMTc4NGY0IiwidCI6IjBmZWQw\
        M2EzLTQwMmQtNDYzMy1hOGNkLThiMzA4ODIyMjUzZSIsImMiOjEwfQ%3D%3D&embedImagePlaceholder=true&pageName=ReportSection" frameborder="0" allowFullScreen="true"></iframe>', unsafe_allow_html = True)

    elif page == "Sentiment Analysis":
      st.title("Sentiment Analysis 😊🙁")
      sent_choice = st.selectbox("Choose one:", ["What is Sentiment Analysis?", "Sentiment Analysis Results"])

      if sent_choice == 'What is Sentiment Analysis?':
        
        st.write("Sentiment analysis (also known as opinion mining) is a natural language processing (NLP) approach for determining the positivity, negativity, or neutrality of data. \
        Sentiment analysis is frequently used on textual data to assist organisations in tracking brand and product sentiment in consumer feedback and better understanding customer demands.")
        image = Image.open("/content/drive/MyDrive/Colab Notebooks/Hotel_Analysis/Images/SA.png")
        st.image(image, caption='Sentiment Analysis')
        
        with st.expander("Definition of CountVectorizer (CV) & Term frequency-inverse document frequency (TF-IDF)"):
          st.write("**CountVectorizer (CV)** is used to convert a collection of text documents to a vector of term/token counts. ")     
          st.write("Similarly, **TF-IDF (term frequency-inverse document frequency)** is a statistic formula that examines the relevance of a word to a document in a collection of documents.\
                    This is accomplished by multiplying two metrics: the number of times a word appears in a document and the word's inverse document frequency over a collection of documents.")
          
          st.write("Let's have a look at what are the differences between CV and TF-IDF using this example below. ")
          st.text("""
          Text1 = “Natural Language Processing is a subfield of AI”\n
          tag1 = "NLP"\n
          Text2 = “Computer Vision is a subfield of AI”\n
          tag2 = "CV
          """)

          st.write("When the two texts are converted into count frequency using **CountVectorizer**, the output will be as follows:- ")
          image = Image.open("/content/drive/MyDrive/Colab Notebooks/Hotel_Analysis/Images/CV output.png")
          st.image(image, caption='CV')

          st.write("On the other hand, the above two texts can be also be converted into term frequency-inverse document frequency using **TFIDF**. The output will be as follows:- ")
          image = Image.open("/content/drive/MyDrive/Colab Notebooks/Hotel_Analysis/Images/TFIDF.png")
          st.image(image, caption='TF-IDF')

          st.write("Reference: https://www.linkedin.com/pulse/count-vectorizers-vs-tfidf-natural-language-processing-sheel-saket/")

      elif sent_choice == 'Sentiment Analysis Results':
        image = Image.open("/content/drive/MyDrive/Colab Notebooks/Hotel_Analysis/Images/Sent Analysis.png")
        st.image(image, caption='Sentiment Analysis (Results Obtained)')

        st.subheader("Explanation")
        st.write("In this project, the end goal is to build a sentiment classification model that can properly tag the reviews into their respective sentiment classes (either Positive / Negative).")
        st.write("A total of 8 models were built, whereby each of their results were compared and evaluated thereafter.")
        st.write("**As a result, the Logistic Regression (LR) Model using the TF-IDF approach managed to receive the highest accuracy of 94%. Hence, we have selected this model as the best model built.**")
        st.write("Feel free to play around with the sentiment classifier which was built using this LR model, as well as taking a closer look at the model's results. :)")
        sent_choice = st.radio("Choose one operation:", ["Sentiment Classifier", "Sentiment Model Results", "Word Cloud Visualisation"])
        
        if sent_choice == "Sentiment Classifier":
            # Sentiment Classification
              st.header("Sentiment Classification")
              sent_model = pickle.load(open('/content/drive/MyDrive/Colab Notebooks/Hotel_Analysis/Sentiment Analysis/Sentiment Models/LR_SentAnalysis.sav' , 'rb'))
              user_input = st.text_area("Enter a review", "i loved the trip!")
              if st.button('PREDICT ▶️'):
                  a = sent_model.predict([user_input])[0]
                  b = sent_model.predict_proba([user_input])[0]

                  st.write("Sentiment Predicted: ")
                  st.subheader(f"**{a}**")
                  st.write("Sentiment Score [Neg vs Pos probability]: ")
                  st.subheader(f"**{b}**")
                  st.balloons()
        
        elif sent_choice == "Sentiment Model Results":
            data = load_data_cleaned()
            sent_model = pickle.load(open('/content/drive/MyDrive/Colab Notebooks/Hotel_Analysis/Sentiment Analysis/Sentiment Models/LR_SentAnalysis_IMPROVED.sav' , 'rb'))
            
            # Defining predictor & Target variable
            X = data['cleaned_Reviews'] # predictor variable
            y = data['Sentiments'] # target variable
            
            # Split dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state= seed)

            
            result_choice = st.selectbox("Choose one performance metrics:", ["Accuracy, F1-score", "Classification Table", "Confusion Matrix"])
        
            if result_choice == "Accuracy, F1-score":
                  predictions = sent_model.predict(X_test)   
                  st.write("Accuracy : " , round((accuracy_score(y_test, predictions)*100),2))
                  st.write("F1 score : " , round(f1_score(y_test, predictions, average='weighted'), 3))
                  
                  with st.expander("Accuracy Definition"):
                    st.write("""
            **Accuracy** is the most intuitive performance measure and it is simply a ratio of correctly predicted observation to the total observations. 

            Accuracy is a great measure but only when the dataset is symmetrical (where values of false positive and false negatives are almost same). 
            Therefore, it is good to also look at other parameters to evaluate the performance the model. For our sentiment classification model, we have got 0.939 which means our model is approx. 94% accurate.
            """)

                  with st.expander("Precision Definition"):
                    st.write("""
            **Precision**  is the ratio of correctly predicted positive observations to the total predicted positive observations. \
             The question that this metric answer is of all reviews labelled as positive, how many are actually positive? High precision relates to the low false positive rate. We have got 0.94 precision which is pretty good.
            """)

            elif result_choice == "Classification Table":
                  classification_table(data, sent_model, X_test, y_test)

                  with st.expander("Recall Definition"):
                    st.write("""
            **Recall** is the ratio of correctly predicted positive observations to the all observations in the actual positive class. \
            The question recall answers is: Of all the positive reviews,how many did we label correctly? We have got recall of 0.94 which is good for this model.
            """)

                  with st.expander("F1-score Definition"):
                    st.write("""
            **F1-score** is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. \
            F1 is usually more useful than accuracy, especially when uneven class distribution exists. In our case, F1 score is 0.94.
            """)

              
            elif result_choice == "Confusion Matrix":
                  confusion_matrix(data, sent_model, X_test, y_test)

                  with st.expander("Confusion Matrix Definition"):
                    st.write("""
            **Confusion Matrix** is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known. The following are the components of a Confusion Matrix: -

                a. True Positives (TP) - These are the correctly predicted positive values which means that the value of actual class is yes and the value of predicted class is also yes. E.g. if actual class value indicates that the review is positive and predicted class tells you the same thing.

                b. True Negatives (TN) - These are the correctly predicted negative values which means that the value of actual class is no and value of predicted class is also no. E.g. if actual class says this review is positive and predicted class tells you the same thing.

                c. False Positives (FP) – When actual class is no and predicted class is yes. E.g. if actual class says this review is negative but predicted class tells you that the review is positive.

                d. False Negatives (FN) – When actual class is yes but predicted class in no. E.g. if actual class value indicates that the review is positive but the predicted class tells you that review is negative.
              """)
                  

        elif sent_choice == "Word Cloud Visualisation":
            # Word Cloud Visualisation
              data = load_data_cleaned()
              st.header("Word Cloud Visualisation")
              option = st.selectbox("Which Sentiment to Display?", ["Positive", "Negative"])
              
              if option == "Positive":
                    st.subheader("Word Cloud for Positive Reviews 😊")
                    positivedata = data[data['Sentiments'] == "Positive"]
                    positivedata =positivedata['cleaned_Reviews']
                    st.write("Positive words are as follows: -")
                    with st.spinner("Generating Wordcloud"):
                      wordcloud_draw(positivedata,'white')

                  
              elif option == "Negative":
                    st.subheader("Word Cloud for Negative Reviews 🙁")
                    negativedata = data[data['Sentiments'] == "Negative"]
                    negativedata =negativedata['cleaned_Reviews']
                    st.write("Negative words are as follows: -")
                    with st.spinner("Generating Wordcloud"):
                      wordcloud_draw(negativedata,'white')

              
    elif page == "Topic Modelling":
      lda_model = gensim.models.ldamodel.LdaModel.load('/content/drive/MyDrive/Colab Notebooks/Hotel_Analysis/Topic Modelling/Topic Models/LDAmallet_NOUNS')
      # Show Topics
      st.title("Topic Modelling 💬")
      topic_choice = st.selectbox("Choose one:", ["What is Topic Modelling?", "Topic Modelling Results"])
      
      if topic_choice == "What is Topic Modelling?": 
      
       st.subheader("What is LDA?")
       st.write("""
          **Latent Dirichlet Allocation (LDA)** is a popular topic modelling approach to extract themes from a corpus. 
          The phrase "latent" refers to something that is there but not fully formed. In other terms, latent refers to something that is hidden or concealed.
          The themes we'd want to extract from the data are now "hidden topics". It has yet to be found. As a result, the term "latent" is used in LDA. 
          
          The challenge of Topic Modelling is how to extract good quality of topics that are clear, segregated and meaningful. This depends heavily on the quality of text preprocessing and the strategy of finding the optimal number of topics.\n
          
          **Two fundamental assumptions are made by the LDA:**\n
          1. Documents are made up of a variety of subjects (mixture of topics)\n
          2. Each topic are made up of a number of tokens (or words) 
          """)
       image = Image.open("/content/drive/MyDrive/Colab Notebooks/Hotel_Analysis/Images/LDA-algorithm.png")
       st.image(image, caption='LDA Algorithm')
       st.write("""
          Reference:\n
          https://www.analyticsvidhya.com/blog/2021/06/part-2-topic-modeling-and-latent-dirichlet-allocation-lda-using-gensim-and-sklearn/#:~:text=Latent%20Dirichlet%20Allocation%20(LDA)%20is,are%20also%20%E2%80%9Chidden%20topics%E2%80%9D.
          """) 
          
       st.subheader("Evaluation Metrics used for Topic Modelling in this project")
       st.write("""
            **Coherence Score** - The interpredability of the topic model. The higher the better. For our model, we managed to obtain a coherence score of 0.511.\n

            **Minimize Topic Overlapping** - Ideally, the topic model should have less topic overlapping (multiple topics talking about the same subject). Also, the further the topics in the Inter topic Distance Map visualisation, the better. 
            """)
       st.write("Reference: https://namyalg.medium.com/how-many-topics-4b1095510d0e") 
       
       
      elif topic_choice == "Topic Modelling Results":
        image = Image.open("/content/drive/MyDrive/Colab Notebooks/Hotel_Analysis/Images/Topic Model.png")
        st.image(image, caption='Topic Modelling (Results Obtained)')

        st.subheader("Explanation")
        st.write("In this project, the end goal is to build a topic model that can help us to identify the most talked subjects in the reviews.")
        st.write("A total of 4 models were built, whereby each of their results were compared and evaluated thereafter.")
        st.write("**As a result, the LDA Mallet Model using the NGRAMS + NOUNS approach was chosen as the best model built, based on 2 evaluation metrics (coherence score + intertopic distance)**. \
        Although its coherence score is only the second highest of the 4 models, its intertopic distance shows that its topics are widely distributed, making it useful for us to identify diverse topics mentioned in the reviews.")
        st.write("Feel free to have a look at the keywords extracted using the best model built, by exploring each of the word cloud visuals, as well as taking a closer look at the interpretations of the topics :)")
      
        st.header("Topic Keywords")

        choice = st.multiselect("Pick Number of Topics to view", range(lda_model.num_topics))
        all_options = st.checkbox("Select all options")
        if all_options:
          choice = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

        for t in choice:
          plt.figure()
          plt.imshow(WordCloud(background_color='white').fit_words(dict(lda_model.show_topic(t, 200))))
          plt.axis("off")
          plt.title("Topic #" + str(t))
          st.pyplot()

        with st.expander("Topics Identified"):
          st.write("""
          The main **disadvantage** of topic modelling is the need to interpret the topics ourselves. \
          Although the model will display us the keywords present in the specific topic, we still need to identify the keywords in each topic, then make our inferences based on the topic words.\n
          Upon interpreting the keywords in each topic, 10 most mentioned topics were identified, as follows: \n
          1. **Room view related** - Keywords: view, balcony, window\n

          2. **Room size / Comfort related** - Keywords: bed, comfy, cozy, cosy, pillow, mattress, uncomfort, size, space, family, luxury, twin, executive\n

          3. **Bathroom related** - Keywords:  bathroom, floor, shower, bath, water, toiletry, sink, slipper\n

          4. **Facility related** - Keywords: facility, furniture, wardrobe, kettle, coffee, milk, tea, pool, facility, fridge, cup, parking\n

          5. **Service related** - Keywords: staff, service, reception, helpful, polite, desk, concierge, efficient, kind, customer, receptionist\n

          6. **Food / Dining related** - Keywords: breakfast, dinner, buffet, choice,fruit , reservation, restaurant, food, cafe, selection, option\n

          7. **Stay Experience related** - Keywords: noise, problem, smell, control, temperature, light, hear, sound, loud, construction, street\n

          8. **Nightlife related** - Keywords: night, drink, lounge, cocktail, bar\n

          9. **Location / Accessiblity related** - Keywords: location, area, station, metro, proximity, walk, tube, tram, train, access, bus, car, convenient, attraction, airport, distance, taxi, public, transport\n

          10. **Internet related** - Keywords: wifi, internet, connect
          """)

        with st.expander("Topics Distribution in Dataset"):
          st.write("**Distribution of Overall Topics in the Dataset**")
          data = load_data_topics()
          df_issues = data.iloc[:, -10:-1]
          k = df_issues.sum()/len(df_issues)
          # k.plot.bar(figsize=(20,10))
          plt.title("Topic Distribution in Reviews")
          st.bar_chart(k)

          st.write("**Distribution of Negative Topics in the Dataset**")
          data = load_data_topics()
          data_neg = data[data['Sentiments'] == "Negative"]
          data_neg = data_neg.iloc[:, -10:-1]
          k = data_neg.sum()/len(data_neg)
          # k.plot.bar(figsize=(20,10))
          plt.title("Negative Topic Distribution in Reviews")
          st.bar_chart(k)

          st.write("**Distribution of Positive Topics in the Dataset**")
          data = load_data_topics()
          data_neg = data[data['Sentiments'] == "Positive"]
          data_neg = data_neg.iloc[:, -10:-1]
          k = data_neg.sum()/len(data_neg)
          # k.plot.bar(figsize=(20,10))
          plt.title("Positive Topic Distribution in Reviews")
          st.bar_chart(k)
          

       
@st.cache
def load_data_ori():
    path = '/content/drive/MyDrive/Colab Notebooks/Hotel_Analysis/Hotel_Datasets/515k_Hotel_Reviews.csv'
    df = pd.read_csv(path)
    return df

def load_data_cleaned():
    path = '/content/drive/MyDrive/Colab Notebooks/Hotel_Analysis/Hotel_Datasets/515k_Hotel_Reviews_SENTIMENTS_40tokens.csv'
    df_clean = pd.read_csv(path)
    return df_clean

def load_data_topics():
    path = '/content/drive/MyDrive/Colab Notebooks/Hotel_Analysis/Hotel_Datasets/515k_Hotel_Reviews_TOPIC_DISTRIBUTION.csv'
    df_topic = pd.read_csv(path)
    return df_topic

def wordcloud_draw(data, color = 'white'):
    words = ''.join(list(data.values))
    wordcloud = WordCloud(
                      background_color=color,
                      max_words=5000,
                     ).generate(words)
    plt.figure(1,figsize=(15, 8))
    plt.imshow(wordcloud)
    plt.axis('off')
    st.pyplot()

def classification_table(data, sent_model, X_test, y_test):
  predictions = sent_model.predict(X_test)
  st.write(classification_report(y_test, predictions))

def confusion_matrix(data, sent_model, X_test, y_test):
  ConfusionMatrixDisplay.from_estimator(sent_model, X_test, y_test)
  st.pyplot()

if __name__ == "__main__":
    main()
