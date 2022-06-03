# Machine Learning Algorithms for Bankruptcy Prediction

In this project I look at financial data for companies listed on Taiwan's stock exchange between 1999 and 2009 and test a series of classification models that use this data to predict which companies are likely to go bankrupt. A bankruptcy prediction model can help investors or fund managers to assess the risk of holding shares of a given company as well as the opportunities and risks of short selling a company's shares. This is purely an educational project so none of the recommendations in it should be construed as financial advice.

## Business Application

There are two main ways that investors can make use of our model:

1. They can use it to pursue a conservative investment strategy by avoiding companies with a high probability of bankruptcy. 
2. They can use it to identify companies that are highly likely to go bankrupt and attempt to profit from shorting the shares of those companies.


## Model Evaluation

I'm going to be evaluating the performance of my models primarily based on the F1 scores for the bankruptcy class, since this number incorporates both:

1. What percentage of all bankruptcies our predictions account for (recall) and
2. What percentage of our bankruptcy predictions are correct (precision).

F1 is defined as:

2 * precision * recall / (precision + recall)

A high f1 score means that our model is both able to achieve our conservative goal of separating bankruptcies from non-bankruptcies as well as our aggressive goal of finding a high precision bankruptcy class. 

## Simplifying Assumptions

In determining potential profits and losses from various investment strategies I'll be making two basic assumptions:

1. Bankruptcy reduces the value of a company's shares to zero.
2. The profit from shorting selling the shares of companies that go bankrupt is on average equal to the losses from short selling the shares of companies that don't go bankrupt.

The first assumption is realistic and allows us to calculate expected losses due to bankruptcy. For example, if we are able to predict non-bankruptcies with 99% precision at a certain probability threshold, then we can expect losses of 1%.

While the second assumption is unrealistic, it's useful because it means that if we are able to predict bankruptcies with >50% precision at a given probability threshold, then we can expect that profits will exceed losses if we short the shares of companies whose probability of bankruptcy falls above the threshold. In the real world, our model would have to use additional factors to determine expected profits and losses, but making this assumption is the best way to quantify the risk of shorting strategies given only the information available in our data set.

## The Data

### Class Imbalance

This data set is highly imbalanced with only 3.2% of companies belonging to the bankruptcy class. This means that a model that predicts the majority class 100% of the time will have 96.8% accuracy. If we're trying to pick out non-bankruptcies for a conservative investment strategy, then we want our model to be able to do so with >96.8% precision, since otherwise, we would be just as well off investing randomly in the market without the aid of a model.

### Data Features

#### Scaling

Most of the data set has already been min-max scaled. This makes the data less interpretable, since we don't know true min and max values. The best way to interpet the data is therefore to convert values using a standard scaler, which enables us to discuss the data points in terms of z-scores (standard deviations from the mean). 

Twenty four columns appeared to be inconsistenty min-max scaled. Most of these had max values of around 10 billion, while median values were below 1. I didn't want to eliminate the rows with outlying values because I didn't want to bias the data set against high values, especially since this would likely reduced the minority class represented in the data set still further. I therefore dropped the columns that hadn't been min-maxed scaled.

#### Correlated variables

The data set included two pairs of variables that were perfectly correlated - in one case because they were definitionally identical and in another case because they represented an accounting identity. These variables were eliminated from the model.

There were also around 10 pairs of variables that had above 99.9% correlation. These included Net Value for Class A, Class B, and Class C shares. It makes sense that different classes of shares of the same company would typically move up and down together. However, because owners of different classes of shares may fare differently in the event of a bankruptcy, I chose not to remove these variables from the model. Tree models would almost certainly only make use of one of these variables as would a Logistic Regression Model making use of Regularization.

## Models Used

### Baseline Model: Depth 2 Decision Tree:

Our baseline model is a decision tree algorithm. It divides the test data into a bankruptcy class and non-bankruptcy class by choosing a splitting criterion that minimizes the entropy within each class. An entropy of 1 means that a node of the tree is split evenly between classes, while an entropy of 0 means that the node is composed of only a single class. Because an unweighted model would give us close to zero entropy at the tree's root, I scaled the minority class by a balancing factor to give the root node an entropy of zero. 

After dividing the data into a bankruptcy class and a non-bankruptcy class, the tree also divided each of these classes into low entropy and high entropy leaves.

![Baseline Decision Tree](images/dt1_tree.png)

Below we can see how this decision tree performed on the testing data:

![Baseline Confusion](images/dt1_confusion.png)

With an F1 score of 0.17, the baseline model successfully picked out 49 out of 58 bankruptcies, but it also yielded 456 false positives. With 99% precision at picking out non-bankruptcies, it's quite useful for giving us a conservative investment strategy, but useless for identifying companies whose shares we might want to short.

Looking at the histograms below, we can see why the decision tree's first splitting criterion - the Net Income to Total Assets Ratio - is so effective at filtering out bankruptcies:

![Decision Tree Histogram](images/dt1_hist.png)

The overwhelming majority of bankrupt companies fall below the Income Asset Ratio decision threshold, while non-bankrupt companies are somewhat more likely to fall above the threshold. This 'somewhat more likely' is the key problem with our baseline model and the reason why our F1 score is so low.

### Logistic Regression Model

