import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import nltk
from nltk.tokenize import word_tokenize
import wordcloud
from wordcloud import WordCloud

from PIL import Image

st.set_page_config(
    page_title='Sentiment Analysis of Amazon Customer Reviews',
    layout = 'wide',
    initial_sidebar_state='expanded'
)

def run():
    # title
    st.title('Sentiment Analysis of Amazon Customer Reviews')
    st.write('by Ahmad Luay Adnani')

    # sub header
    st.subheader ('Exploratory Data Analysis of the Dataset.')

    # Add Image
    image = Image.open('amazon.jpg')
    st.image(image,caption = 'illustration')

    # Description
    st.write('Amazon is one of the largest e-commerce platforms in the world, with millions of products available for purchase and billions of reviews submitted by customers. The reviews can vary in length, tone, and style, and often contain sarcasm, irony, or other forms of nuanced language. Sentiment analysis on Amazon reviews can **provide valuable insights for businesses**, including identifying common issues that customers face with their products or services, understanding the factors that drive customer satisfaction, and tracking changes in customer sentiment over time.')
    st.write('# Dataset') 
    st.write('Dataset used is amazon review dataset from [kaggle]("https://www.kaggle.com/datasets/yacharki/amazon-reviews-for-sa-binary-negative-positive-csv").')

    # show dataframe
    df2 = pd.read_csv('dataset_50000_rows.csv')
    df2 = df2.drop(['Unnamed: 0'],axis=1)
    st.dataframe(df2)
    # add description of Dataset
    st.write('In this dataset, class 1 is the **negative review** and class 2 is the **positive review**')

    ###
    # create a copy of the dataframe
    df_eda = df2.copy()
    df_eda.class_index.replace({1:'Negative Review',2:'Positive Review'}, inplace=True)
    # Separating positive & negative review
    positive_review = df_eda[df_eda['class_index']=='Positive Review']
    negative_review = df_eda[df_eda['class_index']=='Negative Review']

    # Histogram and Boxplot based on user input
    st.write('# Exploratory Data Analysis')
    select_eda = st.selectbox('Select EDA : ', ('Type of Review','Example of Positive and Negative Review','Number of Words','WordCloud'))
    if select_eda == 'Type of Review':
        review = df_eda['class_index'].value_counts().to_frame().reset_index()
        fig = px.pie(review,values='class_index', names='index',color_discrete_sequence=['red','blue'])
        fig.update_layout(title_text = "Type of Review")
        st.plotly_chart(fig)
        st.write('Based on the table and visualization above, it can be seen that both negative and positive reviews consist of 10,000 reviews each.')
    elif select_eda == 'Example of Positive and Negative Review':
       # Print sample reviews
        pd.set_option('display.width', None)
        sample_negative_review = df_eda[df_eda['class_index']=='Negative Review'].sample(20)
        sample_positive_review = df_eda[df_eda['class_index']=='Positive Review'].sample(20)

        # Print Sample of Negative Review
        st.write('Example of Negative Reviews')
        st.write('-'*100)
        for i in range(0,20):
            st.write(sample_negative_review.iloc[i,2])
        st.write('-'*100)

        # Print Sample of Positive Review
        st.write('Example of Positive Reviews')
        st.write('-'*100)
        for i in range(0,20):
            st.write(sample_positive_review.iloc[i,2])
        st.write('-'*100)
        st.write('Based on the examples of various reviews above, what distinguishes between positive reviews and negative reviews :')
        st.write('1. **Positive reviews** generally express satisfaction with a product, service, or experience, highlighting the positive aspects and benefits that were experienced. They often use positive language and may include specific examples of what the reviewer liked about the product or service. They may also mention the quality, value for money, or ease of use.')
        st.write('2. **Negative reviews**, on the other hand, typically express dissatisfaction or disappointment with a product, service, or experience. They often highlight specific problems or issues that the reviewer experienced, such as poor quality, bad customer service, or difficulty using the product. Negative reviews may use negative language and may also include suggestions for how the product or service could be improved.')
        st.write('Overall, the key difference between positive and negative reviews is the attitude of the reviewer and their overall satisfaction with the product or service. Positive reviews reflect a positive experience, while negative reviews reflect a negative experience.')
        


    elif select_eda == 'Number of Words':
       # Count the number of words in each review
        df_eda['len_words'] = df_eda['review_text'].apply(lambda x: len(nltk.word_tokenize(x)))
        # Histogram plot for each review
        fig, ax =plt.subplots(1,2,figsize=(30,10))
        sns.histplot(ax=ax[0],data=df_eda[df_eda['class_index'] == 'Positive Review']['len_words'],kde=True)
        ax[0].set_title('Positive Review')
        sns.histplot(ax=ax[1],data=df_eda[df_eda['class_index'] == 'Negative Review']['len_words'],kde=True)
        ax[1].set_title('Negative Review')
        st.pyplot(fig)
        st.write('-'*100)
        # Print Max and Average number of words

        st.write('The maximum number of words on each review is ', df_eda['len_words'].max())
        st.write('The average number of words on each review is', df_eda['len_words'].mean())
        st.write('-'*100) 
        # Print Max and Average number of words on positive review

        st.write('The maximum number of words on positive review is ', df_eda[df_eda['class_index']=='Positive Review']['len_words'].max())
        st.write('The average number of words on positive review is', df_eda[df_eda['class_index']=='Positive Review']['len_words'].mean())
        st.write('-'*100) 
        # Print Max and Average number of words on negative review

        st.write('The maximum number of words on negative review is ', df_eda[df_eda['class_index']=='Negative Review']['len_words'].max())
        st.write('The average number of words on negative review is', df_eda[df_eda['class_index']=='Negative Review']['len_words'].mean())
        st.write('-'*100) 
        st.write('Based on the information above, it is known that **negative reviews have, on average, more words than positive reviews**.  Based on my assumption, there could be various reasons why negative reviews have, on average, more words than positive reviews. Here are some possible explanations:')
        st.write("1. **Complex issues**: Negative reviews might involve more complex issues or problems with the product or service, which require more detailed explanations and examples.")
        st.write('2. **Emotional expression**: Negative reviews might include more emotional expression, such as frustration or disappointment, which can lead to more detailed and expressive language.')
        st.write('3. **Expectations**: Negative reviews might involve higher expectations from customers, leading them to provide more detailed feedback in order to articulate their disappointment or frustration.')
        st.write('4. **Personal experience**: Negative reviews might be based on a more personal experience, such as a defective product or poor customer service, which can lead to a more detailed and personalized account of the issue.')

    else:
        # Creating wordcloud
        text_positive = positive_review.review_text.values
        cloud_positive = WordCloud(max_words=50, background_color="white",width=2000,height=1000).generate(" ".join(text_positive))

        # Showing wordcloud
        plt.figure(figsize=(15,10))
        plt.axis('off')
        plt.title("Positive Review",fontsize=20)
        plt.imshow(cloud_positive)
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        # Creating wordcloud
        text_negative = negative_review.review_text.values
        cloud_negative = WordCloud(max_words=50, background_color="black",width=2000,height=1000).generate(" ".join(text_negative))

        # Showing wordcloud
        plt.figure(figsize=(15,10))
        plt.axis('off')
        plt.title("Negative Review",fontsize=20)
        plt.imshow(cloud_negative)
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        st.write('From the 2 visualizations above, we can obtain the following information:')
        st.write("1. **Book**, **movie**, **cd** and **album** are the products most frequently reviewed by amazon customers.")
        st.write('2. The most frequent positive words used by amazon customers included **good**, **great**, **love**, **best**, and **easy**. ')
        st.write('3. The most common negative words used by amazon customers included **bad**, **used** and **problem**.')

if __name__ == '__main__':
    run()