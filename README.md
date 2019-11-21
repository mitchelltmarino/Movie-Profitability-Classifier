# Movie-Profitability-Classifier

## Introduction Section

Many movies are produced and never return a profit after being released. For instance, [Disney’s
“Lone Ranger” film lost $ 190 M USD in 2013](https://www.theguardian.com/film/2013/aug/07/the-lone-ranger-film-flop), demonstrating that even the largest corporations in the film making business don’t always get it right. Therefore, there would be great value in using machine learning to predict the success of a film before starting production.

I aimed to provide a PoC solution to this problem by classifying proposed movies as to whether they
will yield a positive return on investment, based on descriptive features and a target budget for
the proposed film.

## Techniques to Solve the Problem

I did some research about previous techniques used to solve this problem. One paper, titled [Using Decision Trees to Characterize and Predict Movie Profitability on the US Market](https://pdfs.semanticscholar.org/e56b/87a8bff32fa5f466fee870abac5e103480ec.pdf), attempted to predict profitability of a movie using a decision tree. They were only able to achieve 76.22% accuracy with their model, leading us to believe that a decision tree may not be not well suited for this problem. One of my criticisms of this paper was that they predicted a film to be profitable if its gross was larger than its production budget, which may be correct by the definition of profitability, but it does not account for non-obvious costs such as marketing.

A second paper, titled [Early Predictions of Movie Success: The Who, What, and When of
Profitability](https://pdfs.semanticscholar.org/6aef/0c24919030a9d96d4a96e0ad851fa6082630.pdf), approached this problem from an interesting angle. The goal of the paper was to
see what type of features could be created to accurately predict the success of a movie. Some of
the features they created included actor gross, director gross, and actor tenure. Using these newly
created features, the classification accuracy of their Random Forest model increased from 74.9%
to 90.4%. I deemed this to be an incredible approach to the problem and have adopted this
approach to create similar features which could be derived from my chosen datasets.

A third paper, [Predicting Movie Success Using Machine Learning Techniques](https://diva-portal.org/smash/get/diva2:1106715/FULLTEXT01.pdf), attempted to use decision tree, SVM, and KNN to predict the revenue category of a film. The paper details the data processing steps they used very well, which I drew inspiration from to categorize data of my own. This was particularly useful when I had to process data of my own. I particularly found how they dealt with genres useful, although genres were not a very strong predictive feature for us in the end.

The final paper, [Machine Learning on Predicting Box Office Gross](http://cs229.stanford.edu/proj2016/report/PengdaLiu-MachineLearningOnPredictingGrossBoxOffice-report.pdf), attempted to
predict the gross of a movie by categorizing gross into 10 bins. They achieved decent success by
using linear regression, state vector machine, and artificial neural network models among others.
This led us to believe that the relationship of descriptive features and profitability can potentially
be captured effectively by using a linear model or ANN. I found error-based learning to be a
somewhat intuitive approach for this problem, which was confirmed to be a useful idea by this
paper, so I ended up implementing logistic regression and state vector machine as two of my
predictive models for this problem.


## Preparing the Data


I chose to use three different datasets for the movie data, most of which included data for the
same movies.

1. [IMDB 5000 Movie Dataset](https://www.kaggle.com/carolzhangdc/imdb-5000-movie-dataset)
    * A dataset of metadata for 5000 different movies, from “Internet Movie Database”.
2. [TMDB 5000 Movie Dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata#tmdb_5000_movies.csv)
    * A dataset of metadata for 5000 different movies, from “The Movie Database”.
3. [TMDB 5000 Credits Dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata#tmdb_5000_credits.csv)
    * A dataset consisting of credits metadata for the “TMDB 5000 Movie Dataset”.


Using Python, Pandas, and NumPy, I performed an inner join on these by the release year and
movie title. Additionally, all films that were not produced in the USA were dropped from the
data which allowed us to keep currency, location, and language consistent among the dataset.
After I was finished combining the data, the dataset consisted of 3,421 instances. I
created and used data quality reports to examine the data properties.

| **Continuous Feature Data Quality Report** | count | miss% | card | min | 25% | 50% | 75% | max |std
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
| **Budget** | 3219 | 0.063 | 343 | 218 | 7500000 | 22000000 | 50000000 | 300000000 | 42497537 
| **Revenue** | 3421 | 0.000 | 2522 | 0 | 14870 | 30749142 | 1.15E+08 | 2787965087 | 176145761
| **Runtime** | 3417 | 0.001 | 151 | 14 | 94 | 104 | 118 | 325 | 21
| **Aspect Ratio** | 3252 | 0.052 | 20 | 1.18 | 1.85 | 1.85 | 2.35 | 16 | 0
| **Full Genre** | 3421 | 0.000 | 726 | 0 | 93 | 179 | 334 | 725 | 177

### Replacing Missing Values and Handling Outliers

By examining the data quality report, I identified that a small quantity of data was missing for
some features. None of the columns were missing data in excess of 1%, so I decided to use
imputation to replace this data. my dataset included some of the top movies of all time, so there
are some very large outliers. Missing values for features with extreme outliers were replaced
with the median value for the feature, otherwise the mean value for the feature was used.

A clamp transformation was later applied to the dataset to remove offending outliers and make
the dataset more representative of common movies. The clamp transformations upper threshold
was Q3 + (1.5 * IQR), and the lower threshold was Q1 - (1.5 * IQR).

**Where:**
- Q1 is the value at the 25% point of the distribution.
- Q3 is the value at the 75% point of the distribution.
- IQR is the Interquartile Range (Q1 – Q3).

### Defining Profitability

In order to a movie to be classified as profitable, it would have a revenue (gross-budget) greater
than zero. However, this definition of profitability does not account for marketing fees, legal
fees, and other costs common to the realm of movie production. Ideally, these additional costs
for making a movie do not exceed the cost of the production budget of the movie.

!["Defining Profitability"](/.Github/Assets/ClassifyingProfitability1.jpg)

Taking these factors into account, I can safely classify a movie as profitable if the revenue of
the movie is greater or equal to the production budget:

After classifying each instance in my dataset, I plotted the class distribution in revenue vs
budget, along with the frequencies which are pictured below.

!["Defining Profitability"](/.Github/Assets/ClassifyingProfitability2.jpg)

### Feature Creation

Based on the research I read, feature creation seemed like a phenomenal way for us to obtain
additional descriptive features for the dataset. I took the following approaches for feature
creation:

- One-hot encoding Genres.
- Extracting the release month from release date.
- Took the number of movies actors and directors had done.
- Took the average revenue of actors, directors, and studios.
- Took the average revenue-budget ratio of actors, directors, and studios.

Afterwards, I ended up with a grand total of 42 descriptive features. A good amount of this
was due to genre, as one-hot encoding genres resulted in an additional 19 descriptive features
alone.

### Feature Selection

Feature selection was an important step of working with my data, because I wanted to have a
good number of informative features while avoiding falling victim to the curse of dimensionality.
I used a combination of box plots and small multiples visualization to determine which
descriptive features were giving us the most information about movie profitability.

!["Defining Profitability"](/.Github/Assets/InformativeAttempt2.jpg)

These findings indicate that the profitability of a movie is highly correlated with the past
successes and activity of directors, actors, and studios. Additionally, the plot of the movie also
plays a role in its potential for success.

## Empirical Evaluation

I implemented four machine models to predict whether a film will be profitable or not. Each
of the models I implemented are well suited for binary classification problems such as
predicting movie profitability.

Before placing the datasets into the machine learning models, I standardized the values into the
range [0, 1]. Additionally, every model was trained and tested by using 10-fold cross validation.

### K Nearest Neighbours (KNN)

The first algorithm I attempted was the K Nearest
Neighbours algorithm, with Euclidean Distance for
the distance metric.

To find the optimal value for K, which I found to
be 22, I iteratively trained and tested the model for
K values 1 to 40.


KNN performed reasonably well considering how simple it is, and it did not require any training
time due to its lazy nature. Thus, it was my fastest trained model.

**K Nearest Neighbours Performance**

!["Defining Profitability"](/.Github/Assets/KNN.jpg)

| Accuracy | Precision | Recall | F-Measure | Area Under Curve
| -- | --- | --- | ---| ---
| 0.89 | 0.88 | 0.93 | .096 | 0.89

### Logistic Regression

The second model I implemented was logistic regression,
which is a linear model made for binary classification, rather
than regression. The goal of a logistic regression model is to
accurately find the optimal decision boundary on the
hyperplane. This is done by using gradient descent to
minimize the sum of squared errors on the training dataset.

my logistic regression model captured the relationship between movie profitability and the
descriptive features quite accurately, but not as well as KNN. The average training time for the
training set (3020 instances) was 7.55ms, making it my second fastest to train model next to
KNN (which requires no training time at all).

**Logistic Regression Performance**

!["Defining Profitability"](/.Github/Assets/LogisticRegressionAUC.jpg)

| Accuracy | Precision | Recall | F-Measure | Area Under Curve
| -- | --- | --- | ---| ---
| 0.87 | 0.88 | 0.90 | 0.89 | 0.87

### Support Vector Machine (SVM)

The third model I implemented was a support vector
machine. The logistic regression model performed reasonably
well, so I intuitively thought that SVM would provide some
greater accuracy due to its inductive bias, which maximizes the
margin of the decision boundary on the hyperplane.

my SVM model performed best with a Gaussian kernel, indicating that the relationship between
movie profitability and the descriptive features was complex. The average training time was
146 ms, making it my second slowest model to train. Interestingly, SVM and KNN had extremely
similar classification results, such that either one could be used depending on preferences of
classification (perhaps true negatives are most important). One may be desirable over the other in
terms of classification or training time, where SVM and KNN would win respectively.

**Support Vector Machine Performance**

!["Defining Profitability"](/.Github/Assets/SVMAUC.jpg)

| Accuracy | Precision | Recall | F-Measure | Area Under Curve
| -- | --- | --- | ---| ---
| 0.89 | 0.91 | 0.91 | 0.91 | 0.89

### Artificial Neural Network (ANN)

The final model I implemented was an artificial neural
network. For my implementation of an artificial neural
network, I experimented with multiple numbers of hidden
layers as well as different sizes of hidden layers. In the end, I
found that an implementation with a single hidden layer with
size 100 provided the best accuracy.

The average training time for the artificial neural network was 5.51 seconds, making it my
slowest model to train, but overall it was the best performing model. This is likely due to the
model’s ability to capture extremely complex relationships between the descriptive feature and
target feature through back propagation.


**Artificial Neural Network Performance**

!["Defining Profitability"](/.Github/Assets/ANNAUC.jpg)

| Accuracy | Precision | Recall | F-Measure | Area Under Curve
| -- | --- | --- | ---| ---
| 0.9 | 0.93 | 0.92 | 0.92 | 0.91



## Conclusion

Although I was only able to achieve a classification accuracy of 91%, it is clearly seen
that machine learning algorithms can make reasonable predictions about the ability of a movie to
turn a profit. Had my data included more descriptive features such as actor rating, which was
heavily correlated with movie success in some of the research papers I read, I would have
been likely able to produce more accurate models.

From the graphs pictured above, it can clearly be seen that the Artificial Neural Network was the
best performing model overall, followed by SVM, KNN, and Logistic Regression.

### Suggestions for Future Research

Because success of a film seems to be heavily correlated with success of past actors, directors,
and studios, I suggest that more information be gathered about these entities both inside and
outside of the domain of movies. Surely social media information, ratings, and awards received
by the film crew would great impact on the ability of a model to predict film profitability.
