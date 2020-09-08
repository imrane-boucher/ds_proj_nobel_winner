# Predictor of Nobel Prize in Economics winning probability

## Background:

Being a student in economics, I enjoy reading papers, books, and articles written by renowned economists. In the course of my studies, I often wondered what truly differentiated
Nobel Prize economists from others. This project was intended to provide an answer to that question, as well as building a tool that could help evaluate economists on their academic records. 
This classifier estimates how likely an economist is to be awarded a Nobel prize during its carrier based on academic performance. As a result, it could for instance be used by the Nobel Foundation as a way to filter economists in their selection process.  

## Project Overview:
* Built a classifier able to identify economists likely to become Nobel Prize winners based on academic performance. The classifier was trained to
correctly recognize true Nobel prize winners **(recall score: 88%)** using a random forest model, so it could then identify correctly economists likely to be nobelized in the future (that is to say economists whose academic performance was comparable to current nobelized economists).
* Scraped over 2000 academic records of economists from the [IDEAS RePEc project](https://ideas.repec.org/top/top.person.alldetail.html) and various Wikipedia pages using beautifulsoup
* Engineered features from the description of each economist to gain valuable insights on their profile (working country, affiliate university ...)
* Optimized Cost-sensitive Logistic Regression and Random Forest Classifier with random undersampling using GridsearchCV to build the best model.
* Used custom evaluation metrics (F2 score, recall) suited for unbalanced datasets (7.5% of positive class in the sample data)
* Created a data app to display the result in a fashionable way using streamlit 

## Code and Ressources used:

**Python version**: 3.7

**Packages**: ```Pandas, NumPy, sickit-learn, matplotlib, seaborn, beautifulsoup, pickle, streamlit, re```

**Project structure (amazing youtube series from Ken Jee)**: https://www.youtube.com/playlist?list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t

**web scraping tutorial youtube**: https://www.youtube.com/watch?v=GjKQ6V_ViQE

**model buidling & fitting unbalanced datasets**: https://machinelearningmastery.com/

**streamlit app**: https://github.com/dataprofessor/code/blob/master/streamlit/part3/penguins-app.py

## Web Scraping:

Used beautiful soup to scrape over 2000 economists' academic record. For each economist I gathered the following features:

+ *nb_downl* (Number of Downloads through RePEc Services over the past 12 months)
+ *nb_pages* (Number of Journal Pages)
+ *Students* (Record of graduates)
+ *nb_works* (Number of works published by the economist)
+ *h_index* (economist scores h when having published h papers that have each been cited at least h times)
+ *nb_cit* (Number of citations)
+ *vn_award* (1 if the economist has been a recipient of the John von Neumann Award else 0)
+ *clark* (1 if the economist has been a recipient of the John Bates Clark Medal else 0)
+ *nobel* (1 if the economist has been nobelized else 0)
+ *top10_shangai_yn* (1 if the economist is affiliated to a university ranked in top 10 of the 2019 Shangai university ranking in the field of economics else 0)
+ *usa_yn* (1 if the economist is currently working in the US else 0)
+ *descri_len* (length in words of the economist's description)*
+ *len_work* (average number of pages of the work published by the economist) 

## Data Cleaning:

Once the data scraped, it took a few steps to clean and prepare the data for modeling:
* merged into one dataframe all the data collected and check for duplicate rows
* check for null values 
* engineer columns for John Bates clark medal, the Von Neumann award, the affiliated university and the working country of the economist
* narrow down the data collected to 400 rows to reduce the level of unbalance of the data
* engineer new features description length and length of work from the data collected

## EDA:

![alt text](https://github.com/imrane-boucher/ds_proj_nobel_winner/blob/master/images/correlation_map.png)
![alt text](https://github.com/imrane-boucher/ds_proj_nobel_winner/blob/master/images/nb_cit_barchart.png)
![alt text](https://github.com/imrane-boucher/ds_proj_nobel_winner/blob/master/images/usa_yn_violin.png)

**Main findings**:
* Nobel prize winners affiliated to a top 10 Shangai university have on average more downloads and more citations, interestingly they also seem to publish less work. Nobel prize winners also score higher h indexes than regular economists. Overall nobelization seems to require strong academic performance and pair recognition.
* According to the violin plot, it seems that on average regular economists (non-nobel winners) working in the us seem to have higher academic performance (h index) than others. For Nobel winners, no clear pattern is reflected from the violin plot

## Model Building:

I split the data into a training and test set and made sure to maintain an equal proportion of each class in bot sets. 
I trained three different models and cross-evaluated them with the ROC score since the Accuracy metric isn't very adapted for highly unbalanced datasets.

three models:
- **KNN Classifier: Baseline for the model** (no tuning to adapt to the unbalanced dataset, used as a baseline)
- **Cost-sensitive Logistic Regression** (tuned the model to adapt for data unbalance)
- **Random Forest with random undersampling** (again tuned the model to adapt for data unbalance / RF can fit complex non-linear classification problems / I had a lot of
y/n features ex: usa_yn, top10_shangai_yn etc.)

## Model Performance:

The Random Forest with random undersampling outperformed the two other models. The primary metric I relied on was **recall** as the first aim of my model was to correctly recognize the largest number of Nobel prize winners so it could then classify positively non yet Nobel promising economists (so the model was intended to keep a loose precision which explains the modest F2 score)

**Performance of the Random Forest with random undersampling on the test set**:

| Recall        | AUC score     | F2 score  |
| ------------- |:-------------:| -----:|
| 0.88           | 0.872         | 0.53 |


## Productionization:

I built a data app with python and streamlit hosted on a local webserver using the youtube tutorial of the data professor referenced above. The app enables the user to test the winning probability of Nobel prize for any economist.

![alt text](https://github.com/imrane-boucher/ds_proj_nobel_winner/blob/master/images/streamlit_eco_app.png)

## Further work:

* I am planning on trying some mean normalization and feature scalling to see if it can improve the model
* I could tune a bit more the model in order to seek for model improvement (the goal would be to improve a bit the precision of the algorithm without losing any recall)
* pay attention to outliers to see if it can improve the model
