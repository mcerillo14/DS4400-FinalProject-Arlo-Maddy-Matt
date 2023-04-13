# Predicting Customer Behavior Using Different Machine Learning Models 

## Abstract:
This repository contains code, data, and step by step directions on how to implement the code. We created different types of models to train our data and compared the accuracy scores of each one at the end. Our data contains instances of customers entering a website that tracks if they purchase something or not. It also keeps tracks of the reason for entering a website, if it was during a weekend/holiday, and many other factors that we could use to predict whether a customer will make a purchase on the website. We implemented a decision tree model, random forest model, logisitic regression, KNN, Naive Bayes, Stochastic Gradient Descent, and a Support Vector model. We thought these models gave us the best range of options to predict our data. In the end, we were able to look at all the accuracy scores for each model which allows us to choose the best method to predict our dataset. 


## Introduction:
The main reason for our project was to try and be able to predict whether a customer will make a purchase when they enter a website. Many people have turned to online shopping instead of going in person due to covid concerns, accesibility, and the increased efficientcy of online vendors. As a result, many compnaies have focused on their online business instead of in person market. So, being able to look at what factors contribute to a customer making a purchase online can be very important to these companies who are trying to succeed in the online market. 

For our approach, we wanted to take a look at a variety of different models to make sure we are getting the best representation of the data. Since we do not if our data is linear or non-linear, this approach allows to find out what type our data is based on which model works the best. Once we created our models, our next step was to look at the initial accuracy score for each and then change the hyperparameters for the models to try and get the best accuracy score.

## Results:
We found accuracy scores and f1 scores for each model and used that to determine which model best predicts our data. Even though both metrics meausre accuracy, we decided to calculate both metrics as accuracy takes into account true positives and negatives, while f1 score takes into account false positives and negatives. So in our decision to determine which model(s) are the best for our data. After going through all of our processes, we detemined the two models with the highest accuracy are Randomn Forest with a 92% accuracy rate and Logistic Regression with a 88% accuracy rate, while the two models with the higest f1 score is  Random Forest with 67% and Decision Tree with 55%. We also found that all of the models have at least an 84% accuracy rate but two of the model only have a less than 15% f1-score. 

To maximize our efficiency for finding the best hyperparameters, we used GridSearch, which is a technique for tuning these parameters until the ones with the best accuracy score. You first have to set your parameter grid with the specifications you want and then the GridSearch function will do the work for you. The first two models we used this strategy for were the decision tree model and random forest model. We trained the models on different max depths as well as different minimum sample splits, which are the minimum number of samples needed to split an internal node. We created before and after results for these two models so we could see how big of an impact tuning the hyperparameters made on our models. For the decision tree model, it started out at an 86% and improved to a 91% with the hyperparameters, while for 

## Discussion:


## Conlusion:


## References:


