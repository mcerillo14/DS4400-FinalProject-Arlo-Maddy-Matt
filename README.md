# Predicting Customer Behavior Using Machine Learning

## Abstract:
We created various machine learning models to predict consumer purchases on a website. Our data contains instances of customers entering a website that tracks if they purchase something or not, the reason for entering a website, if it was during a weekend/holiday, and many other factors. We implemented many different machine learning models to find how we could best predict consumer purchases. 


## Introduction:
The main reason for our project was to try and be able to predict whether a customer will make a purchase when they enter a website. Many people have turned to online shopping instead of going in person due to covid concerns, accesibility, and the increased efficiency of online vendors. Companies want to avoid "online window shopping" as much as possible, where potential customers browse their sites without purchasing anything. Being able to look at what factors are most influential to a customer making a purchase online can be very important to these companies who are trying to succeed in an online market. 

For our approach, we wanted to take a look at a variety of different models to make sure we are getting the best representation of the data. Since we do not know if our data is linear or non-linear, this approach allows to find out what type our data is based on which model works the best. Our data is heavily weighted with thousands of rows of non-purchase visits, and only a few thousand visits that end in purchases. As a result, our models, training, and scoring methods must be able to account for the nature of the data. We can expect a majoirty of our models to score very highly in accuracy, but that will not accurately reflect a model's success. For example, if our model predicts almost all of the non-purchase visits correctly, and very few of the purchase visits correctly, the model accuracy will not reflect the scores we care about. Succesful models for this application will reflect models that predict "true positives" well, not ones that predict "true negatives" well. There are specific guidelines for selecting which classification model depending on the data being classified, as can be seen here: https://blogs.sas.com/content/subconsciousmusings/2020/12/09/machine-learning-algorithm-use/ We are planning to use most of the classification models we learned in this course to best reflect our knowledge and skills developed over the semester, and see which model performs best on our data. Our data is "imbalanced" so we separate it into two classes, the majority class, or in our case, visits that did not end in a purchase. Our minority class are visits that did end in a purchase. 

When evaluating model performance, it is crucial to understand the difference between accuracy, precision, and recall, and how they apply to our use case. Accuracy, the most common metric to evaluate classification models, is defined as the number of predictions the model got right. This is calculated as the number of correct predictions divided by the total number of predictions. Almost all of our models will likely have high accuracy, since they will get the majority of non-purchase visits correctly, which do not matter in our use case. As such, we will want to look at precision and recall to best evaluate model performance. 

Precision, in our use case, will be the number of predicted purchase visits that were actually true purchase visits. It is calculated as the number of true purchases divided by the number of purchase predictions. Precision is also defined as the accuracy score of the minority class (the purchases). Recall, in our use case, is the proportion of actual purchases that were predicted correctly. It is found as the number of predicted true purchases divided by the number of total true purchases. Precision represents the proportion of time a model is correct when it predicts a purchase. Recall represents the proporiton of purchases that are accuractely predicted by the model. F1 score balances this tradeoff well, accounting for both precision and recall when evaluating model performance. We will most prominently emphasize F1 score in evaluating our models, but recall indiviudally is most likely more important to us than precision. We do not care as much if we get our predictions incorrect, but we do care if we miss potential purchases. Imagine if our models were used to identify customers who are likely to purchase something, and then could be given more specific ads, better discounts and promotional offers, or other special treatment. We do not care nearly as much if we give customers who aren't likely to purchase different sets of ads or promtional offers, but potentially missing out on customers who are likely to purchase is a larger drawback. 

## Setup: 
Our dataset consists of 12,331 rows, each representing a visit to a website. After data cleaning, our dataset consisted of 11,958 visits. Out of those visits, 1,863 of them were recorded as visits ending in a purchase. This means our data follows an approximate 85 / 15 split of non-purchases to purchases. The models we are planning to use include Decision Trees, Random Forests, Logistic Regression, KNN, Naive Bayes, Stochastic Gradient Descent, and Support Vector Classification. We plan to use GridSearchCV to adjust our hyperparameters according to paramaters that perform best on training data. 

# NEURAL NETWORKS HERE PLEASE

## Results:
TBD

## Discussion:
TBD 

## Conlusion:
TBD

## References:
1. https://analyticsindiamag.com/7-types-classification-algorithms/
2. https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
3. https://scikit-learn.org/stable/user_guide.html
4. https://en.wikipedia.org/wiki/Receiver_operating_characteristic
5. https://towardsdatascience.com/the-5-classification-evaluation-metrics-you-must-know-aa97784ff226
6. https://blogs.sas.com/content/subconsciousmusings/2020/12/09/machine-learning-algorithm-use/

