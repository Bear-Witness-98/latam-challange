# Challange Notes

## Reviewing the notebook

### Generalities

In the documentation (README) it is said that the data has a column named
DATA-I when talking about the additional DS features, but this column does not
exist. Instead, from the code and the description, we can assume this column
name should be FECHA-I.

### Feedback to the DS

I would like to ask a few things to the DS as it seems unclear why he did some
of the selections he did. I am supposing this is a kind of summary notebook, so
some details may have been missed in the condensation of his/her analysis.

1. I see that at the data splitting step, you decided to keep just three
features. Why keep these features specifically? It seems to me that other
features, as the destination, may encode some information on the delay process;
but maybe I am missing something.
2. On the target encoding, I understood that we wanted to predict the
probability of a delay on a specific flight. Your encoding just predicts whether
the flight would be more than 15 minutes delayed or not. Have you came up with
this threshold from a business perspective, or it was it from some modelling
reason? Have you tried to predict a continuous variable, as delay time, or prob
to be delayed? I know I can get the probabilities from the models, but asking
if you performed an evaluation on them.
3. I would say it's unfair to compare the initial xgboost and linear regression
feature-wise, given their poor initial performance, and the great increase in
performance that happened only by addressing the dataset unbalance. May need a
revision on feature analysis after the balancing or a better exlpanation of
this decision.
4. The 10 features you selected to train with, are not the top 10 I am seeing in
the graph of feature importance from the xgboost. I think it might be due to
some random see issue. Would you care to go over it?

Also pointed out some comments in the code, but wouldn't bother the DS with
them, as it was not the main focus of the work.


### How to continue

I will proceed by moving the pipeline the DS implemented here as similar as
possible to the production pipeline, as this is the best predictions we have
yet. Though, I will also try to make it as versatile as possible in the feature
selection stage, that I think was the one that may need a revision from his
side; though, without compromising the time-to-production of the system.

#### On the model selection
The final model might be modified if another stage of experimenting would be
done by the DS but, as both linear regression and xgboost support the same
interface for prediction, it shouldn't be that much of an issue to change it
afterwards if implemented appropriately.

I will go with the linear regression, as it's execution time is determined only
by the number of features, and not on a highly tunned hyperparameter (as the
number of trees in xgboost). Also, it has the advantage that we can limit
ourselves to only one framework (scikit learn), and have less imcompatibility
issues when trying to move our model to production.

## deployment

application deployed to:  https://delay-model-api-qzo5moezqa-uw.a.run.app