import streamlit as st
import streamlit.components.v1 as components

import numpy as np

import pickle

from PIL import Image

import lime
from lime.lime_text import LimeTextExplainer

def main():

    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.sidebar.title("About")
    st.sidebar.info("Analysis of European Hotel Reviews Dataset")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ["About the Project", "Power BI Dashboard", "Sentiment & Topic Classifier"])

    if page == "About the Project":
        st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:45px;">{"About the Project"}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Problem Statement"}</h1>', unsafe_allow_html=True)
 
        st.write("With the rich amount of information that can be captured through customer reviews, it would be a huge loss if hotel managers chooses to ignore them.\
        By conducting a detailed analysis on the customer reviews, it may present a wider picture about the hotel’s brand, and guarantees that the 'small things' \
        that might boost the hotel’s potential, \
        customer contentment, and customer loyalty are not overlooked.") 
    
        st.write("However, with the huge number of reviews that exists on the Internet, it would be extremely time consuming for hotel managers to \
        analyse and discover anomalies, trends or patterns by going through every single one of them. Hence, the introduction of certain tools, techniques, and expertise \
        would be able to accelerate this time consuming process.") 
        
        st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Dataset Selected"}</h1>', unsafe_allow_html=True)

        st.write("**Link:** https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe")
        
        st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Data Analytics Process"}</h1>', unsafe_allow_html=True)
        image = Image.open("streamlit_template/FYP Data Pipeline (2).png")
        st.image(image, caption='Data Analytics Process')


    elif page == "Power BI Dashboard":
        st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:45px;">{"Power BI Dashboard 📊"}</h1>', unsafe_allow_html=True)
        link='Link to original dashboard: [click here](https://app.powerbi.com/view?r=eyJrIjoiOGM0NDBlYzEtN2RhZS00YjljLTg2NDMtMDY3YTkyM2QzZDg4IiwidCI6IjBmZWQwM2EzLTQwMmQtNDYzMy1hOGNkLThiMzA4ODIyMjUzZSIsImMiOjEwfQ%3D%3D&embedImagePlaceholder=true&pageName=ReportSection)'
        st.markdown(link,unsafe_allow_html=True)
        st.markdown('<iframe title="Hotel (Web version) plus" width="700" height="486" src="https://app.powerbi.com/view?r=eyJrIjoiOGM0NDBlYzEtN2RhZS00YjljLTg2NDMtMDY3YTkyM2QzZDg4IiwidCI6IjBmZWQwM2EzLTQwMmQtNDYzMy1hOGNkLThiMzA4ODIyMjUzZSIsImMiOjEwfQ%3D%3D&embedImagePlaceholder=true&pageName=ReportSection" frameborder="0" allowFullScreen="true"></iframe>', unsafe_allow_html = True)     

     
    elif page == "Sentiment & Topic Classifier":
      st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:45px;">{"Sentiment & Topic Classifier 😄🙁"}</h1>', unsafe_allow_html=True)
        
      sent_model = pickle.load(open('streamlit_template/LR_SentAnalysis_IMPROVED.sav' , 'rb'))
      topic_model = pickle.load(open('streamlit_template/LR_Topic_Label.sav' , 'rb'))
      user_input = st.text_area("Enter a review to predict:", "The check in process was straight forward, the room was very comfortable and clean. The staff were great, and the food was excellent too.")

      if st.button('PREDICT ▶️'):
            a = sent_model.predict([user_input])[0]

            st.markdown(f'<div style="background-color:LightGrey;padding:2px"><h1 style="color:#000000;text-align: center;font-size:24px;">{"Sentiment Predicted:"}</h1>', unsafe_allow_html=True)
            if a == 'Positive':
                st.markdown(f'<h1 style="color:#33ff33;text-align:center;font-size:24px;">{"Positive"}</h1>', unsafe_allow_html=True)
                      
            elif a == 'Negative':
                st.markdown(f'<h1 style="color:#ff0000;text-align:center;font-size:24px;">{"Negative"}</h1>', unsafe_allow_html=True)
                    
            class_names = ['negative', 'positive']
            explainer = LimeTextExplainer(class_names=class_names)
            exp = explainer.explain_instance(user_input, 
                                             sent_model.predict_proba, 
                                             num_features=10)
                  
            st.markdown(f'<div style="background-color:LightGrey;padding:2px"><h1 style="color:#000000;text-align: center;font-size:24px;">{"Most Indicative Words"}</h1>', unsafe_allow_html=True)
            exp.save_to_file('lime.html')
            HtmlFile = open("lime.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read() 
            white_background = "<style>:root {background-color: white;}</style>"
            components.html(white_background + source_code, width=700, height=500, scrolling=True)


            st.markdown(f'<div style="background-color:LightBlue;padding:2px"><h1 style="color:#000000;text-align: center;font-size:24px;">{"Topic(s) Mentioned:"}</h1>', unsafe_allow_html=True)
            y_pred = topic_model.predict([user_input])[0]
            
            topic_names = ['View', 'Comfort/Size',
                                'Bathroom', 'Facility', 'Service',
                                'Food/Dining', 'Stay Experience',
                                'Nightlife', 'Location/Access',
                                'Internet']

            class_labels=[topic_names[i] for i,no in enumerate(y_pred) if no == 1]

            st.markdown(f'<h1 style="color:blue;text-align:center;font-size:24px;">{class_labels}</h1>', unsafe_allow_html=True)
            
            explainer = LimeTextExplainer(class_names=topic_names)
            exp = explainer.explain_instance(user_input, 
                                                topic_model.predict_proba, 
                                                num_features=5, top_labels=5)

            st.markdown(f'<div style="background-color:LightBlue;padding:2px"><h1 style="color:#000000;text-align: center;font-size:24px;">{"Most Indicative Words"}</h1>', unsafe_allow_html=True)
            exp.save_to_file('topic.html', text=False)
            HtmlFile = open("topic.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read() 
            white_background = "<style>:root {background-color: white;}</style>"
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(' ')
            with col2:
                components.html(white_background + source_code, width=450, height=1200,scrolling=True)
            with col3:
                st.write(' ')

            st.balloons()

              
if __name__ == "__main__":
  main()
