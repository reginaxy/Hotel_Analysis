import streamlit as st
import pickle

from matplotlib import pyplot
import matplotlib.pyplot as plt

import gensim
from gensim import corpora
from wordcloud import WordCloud, STOPWORDS

from PIL import Image

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.sidebar.title("About")
    st.sidebar.info("Analysis of European Hotel Reviews Dataset")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ["Homepage", "Power BI Dashboard", "Sentiment Analysis", "Topic Modelling"])

    if page == "Homepage":
        st.title("Objectives of the project")
        st.write("The purpose of this project is to perform an analysis on a European Luxury Hotel Reviews dataset.")
        st.write("**Data Source:** https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe")
        st.write("**Deliverables of the project: -**")
        st.write("1. An interactive Power BI Dashboard to visualize useful findings from the dataset. \n"
                 "2. Sentiment Analysis on the customer reviews, including a Sentiment Classification Model using a suitable Machine Learning (ML) Algorithm. \n"
                 "3. Topic Modelling to discover the most talked topics in the customer reviews.\n")
        
        st.header("Data Pipeline")
        image = Image.open("streamlit_template/Data Pipeline.png")
        st.image(image, caption='Data Pipeline')


    elif page == "Power BI Dashboard":
        st.title("Power BI Dashboard 📊")
        link='Link to original dashboard: [click here](https://app.powerbi.com/view?r=eyJrIjoiZWViNWE4N2YtZmM0OS00MzFkLTk0MDgtMDdlOTZhMTc4NGY0IiwidCI6IjBmZWQwM2EzLTQwMmQtNDYzMy1hOGNkLThiMzA4ODIyMjUzZSIsImMiOjEwfQ%3D%3D&embedImagePlaceholder=true&pageName=ReportSection)'
        st.markdown(link,unsafe_allow_html=True)
#         st.write("Original Dashboard Link: ")
        st.markdown('<iframe title="Hotel (Web version)" width="800" height="486" src="https://app.powerbi.com/view?r=eyJrIjoiZWViNWE4N2YtZmM0OS00MzFkLTk0MDgtMDdlOTZhMTc4NGY0IiwidCI6IjBmZWQw\
        M2EzLTQwMmQtNDYzMy1hOGNkLThiMzA4ODIyMjUzZSIsImMiOjEwfQ%3D%3D&embedImagePlaceholder=true&pageName=ReportSection" frameborder="0" allowFullScreen="true"></iframe>', unsafe_allow_html = True)

    elif page == "Sentiment Analysis":
      st.title("Sentiment Analysis 😊🙁")
        
            # Sentiment Classification
      st.header("Sentiment Classification")
      sent_model = pickle.load(open('streamlit_template/LR_SentAnalysis_IMPROVED.sav' , 'rb'))
      user_input = st.text_area("Enter a review", "I loved the trip!")
      if st.button('PREDICT ▶️'):
                  a = sent_model.predict([user_input])[0]
                  b = sent_model.predict_proba([user_input])[0]

                  st.write("Sentiment Predicted: ")
                  st.subheader(f"**{a}**")
                  st.write("Sentiment Score [Neg vs Pos probability]: ")
                  st.subheader(f"**{b}**")
                  st.balloons()

              
    elif page == "Topic Modelling":
      lda_model = gensim.models.ldamodel.LdaModel.load('streamlit_template/LDAmallet_NOUNS')
      # Show Topics
      st.title("Topic Modelling 💬")
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
