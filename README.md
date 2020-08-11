"# ds_proj_nobel_winner"

# Nobel Prize in Economics winning probability estimator

## Background:

Being a student in economics, I enjoy reading papers, books and articles of renowned economists. A question that I often encountered was what truely differentiates 
Nobel Prize economists from others. This project was intendend to provide a partial answer to that questiion, as well as building a tool that could help evaluate economists on their academic records. 
This classifier estimates how likely an economist is to be awareded a nobel prize during its carrier based on academic performance. As a result it could for instance be used by the Nobel Foundation as a way to filter economists in their selection process.  

## Project Overview:
* Built a classifier enable to differientate economists likely to become Nobel Prize winners, from those who aren't, based on academic performance. The classifier was trained to
correctly recognise true nobel prize winners **(recall score: 88%)** using a random forest model, so it could then identify correctly economists likely to be nobelized in the future (that is to say economists whose academic performance was comparable to current nobelized economists).
* Scraped over 2000 academic records of economists from the [IDEAS Repec project](https://ideas.repec.org/top/top.person.alldetail.html) and various wikipedia pages using beautifulsoup
* Engineered features from the description of each economists to gain valuable insights on their profile (working country, affiliate university ...)
* Optimized Cost-sensitive Logistic Regression and Random Forest Classifier with random undersampling using GridsearchCV to build the best model.
* Created a data app to display the result in a fashionable way using streamlit
