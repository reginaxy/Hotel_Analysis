import streamlit as st
import pickle

from matplotlib import pyplot
import matplotlib.pyplot as plt

import gensim
from gensim import corpora
# import pyLDAvis
# import pyLDAvis.gensim_models
from wordcloud import WordCloud, STOPWORDS

from PIL import Image

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.sidebar.title("About")
    st.sidebar.info("Analysis of European Hotel Reviews Dataset")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ["Homepage", "Power BI Dashboard", "Sentiment Analysis", "Topic Modelling"])

    if page == "Homepage":
        st.title("Introduction of the Project")
        st.header("Purpose")
        st.write("To perform an analysis on a European Luxury Hotel Reviews dataset.")
        st.write("**Data Source:** https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe")
        st.header("Deliverables")
        st.write("1. An interactive Power BI Dashboard to visualize useful findings from the dataset. \n"
                 "2. Sentiment Analysis on the reviews, including a sentiment classifier using a suitable Machine Learning (ML) Algorithm. \n"
                 "3. Topic Modelling to discover the most talked topics in the reviews.\n")
        
        st.header("Data Analytics Process")
        image = Image.open("streamlit_template/Data Process.png")
        st.image(image, caption='Data Analytics Process')


    elif page == "Power BI Dashboard":
        st.title("Power BI Dashboard 📊")
        link='Link to original dashboard: [click here](https://app.powerbi.com/view?r=eyJrIjoiZWViNWE4N2YtZmM0OS00MzFkLTk0MDgtMDdlOTZhMTc4NGY0IiwidCI6IjBmZWQwM2EzLTQwMmQtNDYzMy1hOGNkLThiMzA4ODIyMjUzZSIsImMiOjEwfQ%3D%3D&embedImagePlaceholder=true&pageName=ReportSection)'
        st.markdown(link,unsafe_allow_html=True)
#         st.write("Original Dashboard Link: ")
        st.markdown('<iframe title="Hotel (Web version)" width="800" height="486" src="https://app.powerbi.com/view?r=eyJrIjoiZWViNWE4N2YtZmM0OS00MzFkLTk0MDgtMDdlOTZhMTc4NGY0IiwidCI6IjBmZWQw\
        M2EzLTQwMmQtNDYzMy1hOGNkLThiMzA4ODIyMjUzZSIsImMiOjEwfQ%3D%3D&embedImagePlaceholder=true&pageName=ReportSection" frameborder="0" allowFullScreen="true"></iframe>', unsafe_allow_html = True)

    elif page == "Sentiment Analysis":
      st.title("Sentiment Analysis 😊🙁")
      sent_choice = st.selectbox("Choose one:", ["What is Sentiment Analysis?", "Sentiment Classifier"])
        
      if sent_choice == 'What is Sentiment Analysis?':
        st.header("What is Sentiment Analysis?")
        image = Image.open("streamlit_template/SA.png")
        st.image(image, caption='Sentiment Analysis')
        st.write("Sentiment analysis is a natural language processing (NLP) approach for determining the positivity, negativity, or neutrality of data. \
        Sentiment analysis is frequently used on textual data to assist organizations in tracking brand and product sentiment in consumer feedback and better understand customer demands.")
       
    
      elif sent_choice == 'Sentiment Classifier':
          st.header("Sentiment Classifier")  
          sent_model = pickle.load(open('streamlit_template/LR_SentAnalysis_IMPROVED.sav' , 'rb'))
          user_input = st.text_area("Enter a review to predict", "I loved the trip!")
          if st.button('PREDICT ▶️'):
                  a = sent_model.predict([user_input])[0]
                  b = sent_model.predict_proba([user_input])[0]

                  st.write("Sentiment Predicted: ")
                  st.subheader(f"**{a}**")
                  st.write("Sentiment Score [Neg vs Pos probability]: ")
                  st.subheader(f"**{b}**")
                  st.balloons()

              
    elif page == "Topic Modelling":
       st.title("Topic Modelling 💬")
       topic_choice = st.selectbox("Choose one:", ["What is LDA Topic Modelling?", "Topic Model Results"])
    
       if topic_choice == "What is LDA Topic Modelling?": 
           st.header("What is LDA Topic Modelling?")
           image = Image.open("streamlit_template/LDA-algorithm.png")
           st.image(image, caption='LDA Topic Modelling')
           st.write("""
              Topic modeling is a text processing technique, which is aimed at overcoming information overload by seeking out and demonstrating patterns in textual data, 
              identified as the 'topics'. It enables an improved user experience, allowing analysts to navigate quickly through a corpus of text or a collection, guided by identified topics.\n
              **Latent Dirichlet Allocation (LDA)** is a popular topic modelling approach to extract themes from a corpus. 
              The phrase "latent" refers to something that is hidden or concealed.
              The themes we'd want to extract from the data are "hidden topics". As a result, the term "latent" is used in LDA. 

              The challenges of Topic Modelling is to extract good quality of topics that are clear, segregated and meaningful. \
              This depends heavily on the quality of text preprocessing and the strategy of finding the optimal number of topics.\n

              **Two fundamental assumptions are made by the LDA:**\n
              1. Documents are made up of a variety / mixture of topics \n
              2. Each topic are made up of a number of words
            """)
            
            
       elif topic_choice == "Topic Model Results":
        st.header("Topic Model Results")

        import streamlit.components.v1 as components
#         html_string = ('<h3>streamlit_template/lda.html<h3>')
#         st.components.v1.html(html_string, width=None, height=None, scrolling=False)
        
        HtmlFile = open("streamlit_template/lda.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        print(source_code)
        components.html(source_code, width=1000, height=800, scrolling=True)
        
        lda_model = gensim.models.ldamodel.LdaModel.load('streamlit_template/LDAmallet_NOUNS')
          # Show Topics
        st.header("Topic Keywords")

        num = range(lda_model.num_topics)
        choice = st.multiselect("Pick Number of Topics to view", num)
        all_options = st.checkbox("Select all options")
        if all_options:
           choice = [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
        
        for t in choice:
              plt.figure()
              plt.imshow(WordCloud(background_color='white').fit_words(dict(lda_model.show_topic(t, 200))))
              plt.axis("off")
              plt.title("Topic #" + str(t))
              st.pyplot()

        with st.expander("Topics Identified"):
          st.write("""
          In short, **Topic Modelling** is a text mining approach to find common subjects in texts. \
          Topic modelling can connect words with similar meanings together, \
          and distinguish between usage of words with numerous meanings by assigning them to different topics. \n

          The main **disadvantage** of topic modelling is the need for humans to interpret the topics themselves. \
          Although the model will display the keywords present in each specific topic, humans will still need to identify the most important keywords in each topic, \
          then make logical and reasonable inferences based on the keywords identified.\n

          Upon observing the keywords in all the topics, the 10 most mentioned subjects in the reviews were identified, as follows: \n

          1. **Room size / Comfort related** - Keywords: bed, comfy, cozy, cosy, pillow, mattress, uncomfort, airconditioning, size, space, family, luxury, twin, executive\n

          2. **Bathroom related** - Keywords:  bathroom, floor, shower, bath, water, toiletry, sink, slipper\n

          3. **Room view related** - Keywords: view, balcony\n

          4. **Facility related** - Keywords: facility, furniture, wardrobe, kettle, coffee, milk, tea, pool, facility, fridge, cup, parking\n

          5. **Service related** - Keywords: staff, service, reception, helpful, polite, desk, concierge, efficient, kind, customer, receptionist\n

          6. **Food / Dining related** - Keywords: breakfast, dinner, buffet, choice,fruit , reservation, restaurant, food, cafe, selection, option\n

          7. **Stay Experience related** - Keywords: noise, problem, smell, control, temperature, light, hear, sound, loud, construction, street\n

          8. **Nightlife related** - Keywords: drink, lounge, cocktail, bar, beer\n

          9. **Location / Accessiblity related** - Keywords: location, area, station, metro, proximity, walk, tube, tram, train, access, bus, car, convenient, attraction, airport, distance, taxi, public, transport\n

          10. **Internet related** - Keywords: wifi, internet, connect
          """)

if __name__ == "__main__":
  main()
