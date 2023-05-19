# Sentiment Analysis of Amazon Customer Reviews
This project aims to build, test, and select a Natural Language Processing model that accurately classifies the sentiment expressed in customer reviews by Amazon customers as either positive or negative.

# File Explanation on Github
This repository consists of several files, namely :

- Folder deployment = Contains files used for deployment to HuggingFace (contains models, python applications etc.)
- Notebook_Sentiment_Analysis_of_Amazon_Customer_Reviews.ipynb = This file is the main notebook used to explore dataset and built model
- model_inference_Sentiment_Analysis_of_Amazon_Customer_Reviews.ipynb = Notebook used for testing inference. Inferencing is done on a separate notebook to prove that the model can run on a notebook that is clean of variables
- url.txt = Deployment URL to HuggingFace

# Brief Summary of Project
The flow of this project, first EDA (Exploratory Data Analysis) to find out the basic picture of the dataset. Second, cleaning and preprocessing of the dataset. Third, Built 4 Model (LSTM, Improved LSTM, GRU, Improved GRU) and choose LSTM Improvement as Best Model. Result of the model using Deep Learning with own architecture shows 80% Accuracy on test-set

# Project Conclusion
  - The model we created has 80% accuracy. However, in the inference model, the model we created only achieved 70% accuracy for this dataset. This may be due to the presence of new words that are not included in the model's vocabulary.
  - There are words that appear in both reviews, i.e time, well, little, even, movie,etc. This makes the model misclassified.

