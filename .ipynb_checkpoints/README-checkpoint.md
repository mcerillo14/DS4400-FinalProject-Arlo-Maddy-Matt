# Predicting Customer Behavior Using Machine Learning

## Abstract:
We created various machine learning models to predict consumer purchases on a website. Our data contains instances of customers entering a website that tracks if they purchase something or not, the reason for entering a website, if it was during a weekend/holiday, and many other factors. We implemented many different machine learning models to find how we could best predict consumer purchases. 

## Introduction:
The main reason for our project was seeing how viable and/or possible it is to predict consumer behavior, which could have countless implications in real-world application. Many people have turned to online shopping instead of going in person due to Covid-19 concerns, accesibility, and the increased efficiency of online vendors. Companies want to avoid "online window shopping" as much as possible, where potential customers browse their sites without purchasing anything. Being able to look at what factors are most influential to a customer making a purchase online can be very important to companies who are trying to succeed in an online market. 

For our approach, we wanted to take a look at a variety of different models to make sure we are getting the best representation of the data. Since we did not know if our data is linear or non-linear, this approach allowed to find out what type our data is based on which model works the best. Our data is heavily weighted with thousands of rows of non-purchase visits, and only a few thousand visits that end in purchases. As a result, our models, training, and scoring methods must be able to account for the nature of the data. We expected a majority of our models to score very highly in accuracy, but that will not accurately reflect a model's success. For example, if our model predicted almost all of the non-purchase visits correctly, and very few of the purchase visits correctly, the model accuracy will not reflect the scores we care about. Succesful models for this application will reflect models that predict "true positives" well, not ones that predict "true negatives" well. There are specific guidelines for selecting which classification model depending on the data being classified, as can be seen here: https://blogs.sas.com/content/subconsciousmusings/2020/12/09/machine-learning-algorithm-use/ 

We planned to use most of the classification models we learned in this course to best reflect our knowledge and skills developed over the semester, and see which model performs best on our data. Our data is "imbalanced" so we separate it into two classes, the majority class, or in our case, visits that did not end in a purchase and the minority class, or visits that did end in a purchase. 

When evaluating model performance, it is crucial to understand the difference between accuracy, precision, and recall, and how they apply to our use case. Accuracy, the most common metric to evaluate classification models, is defined as the number of predictions the model predicted correctly. This is calculated as the number of correct predictions divided by the total number of predictions. Almost all of our models will likely have high accuracy, since they will get the majority of non-purchase visits correctly, which do not matter in our use case. As such, we will want to look at precision and recall to best evaluate model performance. 

Precision, in our use case, will be the number of predicted purchase visits that were actually true purchase visits. It is calculated as the number of true purchases divided by the number of purchase predictions. Precision is also defined as the accuracy score of the minority class (the purchases). Recall, in our use case, is the proportion of actual purchases that were predicted correctly. It is found as the number of predicted true purchases divided by the number of total true purchases. 

Precision represents the proportion of time a model is correct when it predicts a purchase. Recall represents the proportion of purchases that are accurately predicted by the model. 

F1 score balances this tradeoff well, accounting for both precision and recall when evaluating model performance. We will most prominently emphasize F1 score in evaluating our models, but recall indiviudally is most likely more important to us than precision. We do not care as much if we get our purchase predictions incorrect, but we do care if we miss potential purchases. One use case is if our models were used to identify customers who are likely to purchase something, and then they could be given more specific ads, better discounts and promotional offers, or other special treatment. We do not care nearly as much if we give customers who aren't likely to purchase different sets of ads or promtional offers, but potentially missing out on customers who are likely to purchase is a larger drawback. 

## Setup: 
Our dataset consists of 12,331 rows, each representing a visit to a website. After data cleaning, our dataset consisted of 11,958 visits. Out of those visits, 1,863 of them were recorded as visits ending in a purchase. This means our data follows an approximate 85 / 15 split of non-purchases to purchases. Our models include Decision Trees, Random Forests, Logistic Regression, KNN, Naive Bayes, Stochastic Gradient Descent, and Support Vector Classification. We used GridSearchCV to adjust our hyperparameters according to the best performing sets on training data.

Our models all follow the same basic structure. The data is split using stratified k-fold cross validation into 80% training data and 20% testing data for the models. Due to the nature of the data, ensuring that the percentages of samples remains representative of the entire dataset is extremely important. For the appropriate models, GridSearchCV is used to find the best set of hyperparameters during training, and that best set is used for the testing data. We utilized F1 scores, recall scores, and confusion matrices to score and evaluate our models. According to the function call utilized in the code, a user chooses which models to call, which matrices to have plotted for them, and how many times to run each model. Due to the random sampling of the training and testing data, scores of each models vary slightly per each run.   

## Results:

#### Main Results:
Due to the nature of our testing data, with under 400 visits in the testing portion of our models, our scores varied considerably from run to run. Over multiple and repeated runs, the models tend to return to their respective "mean" scores, but in one run, some models may score up to .2 higher in F1 score or recall than in others, which is quite significant. Typical ranges for the models are as follows:

| Model | F1 Score | Recall Score |
| :----------- | -----------: | ------------: |
| Decision Tree | .68 - .72 | .55 - .65 |
| Random Forest Classifier | .70 - .74 | .55 - .65 |
| Logistic Regression | .65 - .69 | .45 - .55 |
| KNN | .53 - .57 | .35 - .45 |
| Naive Bayes | .44 - .48 | .45 - .55 |
| Stochastic Gradient Descent | .77 - .81 | .25 - .35 |
| Support Vector Classification | .95 - 1.0 | .10 - .20 |

It is crucial to understand the context of how we are evaluating what a succesful model is to understand which models are effective, and which are not. Despite having two of the highest F1 scores consistently, both Stochastic Gradient Descent and Support Vector models are almost entirely useless for our purposes. The reason they score so highly in F1 is due to their almost perfect precision scores. These models almost never predict purchases, and as a result, see their precision scores skyrocket. We assume this is due to the models only predicting purchases in the most extreme and obvious cases. These models offer us little to no additional insight, and miss the vast majority of purchases in the dataset. This is an example that reinforces why it is so crucial to use recall scoring in our project, with merely F1 score alone we could fairly reasonably conclude that our worst models for our purposes are actually, our best ones. 

The middle tier of model succes include Naive Bayes, KNN, and Logistic Regression models. These models have some of the most irregular and inconsistent scores, but tend to cluster around the reported values in the table. Due to their incosistency, these models should not be relied upon. Occasionally, however, Naive Bayes and Logistic Regression modes have reported F1 scores over .8, and recall scores over .7, which are some of our most succesful models. 

Our most reliable models are the tree-based methods, Decision Trees and Random Forests. As to be expected, our Random Forests outperform our Decision Tree models, but not signifcantly. Both tree-based models have F1 scores around .7, with recall scores centering around .62 or .63. We used GridSearchCV to find the best set of hyperparamters, but often the inputted hyperparameters made little to no improvement on the model, and even in some cases early in our scoring and evaluation process, actually made the models perform worse. 

#### Supplementary Results:
We hypothesized that the nature of our models' scoring was due to the extreme imbalance in the data itself. With approximately 85% of data being in a non-purchase, we wondered if our models would perform better with a more evenly distributed dataset to train on. We took all the purchases, and then randomly selected an equal number of non-purchase visits to create a new dataset, that was exactly half purchases, and half non-purchases to train our models on. We found the new dataset to be inconclusive in changing the models' performance in any notable way. Scores fell into the same ranges, and were neither more or less consistent. 

We used GridSearchCV to make our hyperparameter choices. Due to the nature of the dataset and its imbalanced distribution, we believed the best course of action would be to choose appropriate model parameters based on performance on training data. 

## Discussion:
Accurately predicting consumer behavior, especially online, is no small task. That being said, we expected our models to perform better than they did. The nature of the dataset has been discussed at length, but we believe that one of the main takeaways from our work is that there might merely be no significant correlation between the behavior and purchases. Our dataset comes from one website, in one year, and is no way representative of larger consumer behaviors in other industries, or other time periods, but it does bring up interesting theories about the nature of the task. As we found in our preliminary data exploration, almost all of the data had very little correlation with purchasing factors. 

The other largest conclusion from our work is that we struggled to train models the way we wanted to. Due to the specific nature of our definition of a succesful model, we cared very little about correctly predicting non-purchases, or mistakenly predicting purchases when the visits were actually non-purchases. However, the sklearn .fit() method is somewhat of a black box and we were not able to find a way to train the model to replicate the same attributes we wanted to see. As a result, we were only able to evaluate and measure the models on metrics that reflected our intent, which leads to very low scores. Our models had over 95% accuracy the vast majority of the time, but that metric is not applicable to our real-world scenario, and thus, is irrelevant to the success of our project. 

## Conlusion:
In attempting to predict online consumer behavior, our group had to overcome many challenges and obstacles to create succesful models. We utilized the skills developed over the course of the semester to implement a variety of classification models, using F1 scores and recall as metrics to determine model success. Due to the nature of the imbalanced data, we struggled to effectively train our models to differentiate between purchases and non-purchases, and as a result, our models scored lower than expected. However, the challenges of this project forced us to think creatively and understand the real-world scenario of our project to evaluate, measure, and interpret our results. 

## References:
The Data: 
https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset

Other Sources:
1. https://analyticsindiamag.com/7-types-classification-algorithms/
2. https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
3. https://scikit-learn.org/stable/user_guide.html
4. https://en.wikipedia.org/wiki/Receiver_operating_characteristic
5. https://towardsdatascience.com/the-5-classification-evaluation-metrics-you-must-know-aa97784ff226
6. https://blogs.sas.com/content/subconsciousmusings/2020/12/09/machine-learning-algorithm-use/