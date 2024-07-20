# Challange Notes

Notes during the development of the challange.

## Part I Model selection and transcription

### Reviewing the DS's notebook.

There are quite some issues with how the notebook was presented. The first issue
is that it not runs properly, due to the abscense of the `x=` and `y=` kwargs
missing in the barplots. A second issue is that the `.set` method of seaborn
was deprecated in favour of `.set_theme`; also, this method could only be called
once. And lastly, some subtitles in markdown did not corresponded to the cells
they had below along the First Sight on the data analysis. While not critical,
nor relevant to the result they do not give a good impression of the care taken
while summarising the DS's experiments into this notebook. These errors where
fixed to properly execute the notebook (alongside with the change from
`../data/data.csv` to `data/data.csv`).

#### Data analysis

The colour lightblue is a really poor choice for a contrast between the graph
background and the bars.

The "flight by destination" graph is so cramped up, its hard to see which bar
corresponds to which destination.

#### Features generation

The additional features the DS computed makes sense at first sight, but the
target doesn't match up with what was asked. If the idea was to predict the
**probability** of a flight being delayed or not, then regressor should be
trained, and this information on "how much delay" the flight had, could be
implemented into this encoding. This may have more to do with the modelling part
than with the feature generation, but it is very tightly coupled. Also, the
value of 15 minutes to be considered a delay is not appropriately justified.

#### Data analysis II

First cell of this part shows bad coding practices.

#### Training

`training_data` was defined in second cell but not used again.

The selection of three features is done with no explanation whatsoever. A binary
encong of the `TIPOVUELO` may benefit the model by reducing the input dimension.
a cyclical encoding of the months could benefit the model by: reducing the input
dimension, and, redundantly, adding the cyclicality nature of the months to the
model.

On the xgboost training, the DS added an artificial threshold to its outputs,
this wasn't needed as the model already outputs 0/1. It should also be a red
flag that the model predicted always 0, meaning that it was not capturing enough
the information in the dataset. This result could be used as an argument of the
unbalancing of the dataset (in addition to the Logistic Regression below). By no
means should this trained model used to get the most important features, as it
would take the features that mostly predict the 0 value.

Notice that the top 10 features that are selected, do not match with the top 10
features of the graph (at least in this run).

No mention is done on this whatsoever, a comment on such calling result
should be done.

#### Training II

For a second round of training, the DS considered the most important features to
be more relevant than the balancing, and made the experimentation as such,
getting expected results when the balancing is not done. The results without
the balancing lose meaning in this context.

Though, with the balancing, some better results are obtained. But, there is no
explanation whether this is enough or not for the buisness. I don't think that
should be explicilty in the DS's analysis, but would be a nice to have as a
conclusion of the work.

#### Conclusions and next steps:

I would send back an email to the DS, asking for some more explanations of the
results, and their interpretations. Pointing out some of the mistakes I've
found. Something of the sort:

1. I see that at the data splitting step, you decided to keep just three
features. Why keep these features specifically? It seems to me that other
features, as the destination, may encode some information on the delay process.
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
5. Are these metrics enough from a business perspective? Or they are expected to
be improved on further iterations?

Also pointed out some comments in the code, but wouldn't bother the DS with
them, as it was not the main focus of the work, and I haven't found clear bugs
on the used prediction features, that's what I would use later. Though, I would
report any bugs in the computation of features if I found one.

##### Model selection

At the first iteration of the challange, I choose the LogisticRegression as the
model of choice, for its simplicity, ease of explainability (which feature
it gives more weight to), and because it is part of scikit learn, our training
framework. But given that there are a lot of unsolved mysteries, and we may need
more predictive power, I will now choose the xgboost model. To solve any
additional issue we might have early, and for easiness of retraining. With the
added benefit that we don't loose that much explainability.

##### Feature selection

I will use the features selected by the DS as-is, because it's the best baseline
we have and it does not compromises time-to-delivery on a first iteration.
Putting this list as a parameter and updating it if it needs to isn't too hard.

##### Code-wise

Implemented the functions as similar to what they are in the notebook as
possible, improving as much code as I could.

Added a code to be executed if the script was run with
`python challenge/model.py` to generate and save a trained model in the path:
`models/model.pkl`. Pickle is not the best format to use, due to compatibility
reasons, but as I just saved the LogisticRegression, and used it inside the same
class as it was generated in, I don't expect errors. And converting it to a more
general format (like ONNX) and then running it properly is not trivial.



## Part II API developement

Developed an api to serve the model's predictions properly.

There is a welcome message at the root (`/`) entry-point, a health status check
at the `/health` entry-point, and the prediction service at the `/predict`
entry-point.

This API, expects a directory named `models/` at the level of its execution,
where the model object will look for a `model.pkl` file, which stores a trained
instance of the selected model.

Notice that the api mostly manages the reception of information, and does little
processing, i.e. convert the input list of flights into a pandas dataframe.

Also, while on the prediction stage, where an error may occur, I've decided to
not report the error directly to the client, but to log it in an internal file,
and return a 500 error. This is not scalable, it's just an ad-hoc solution to
unwanted information leak to the client side of the api. Lot more information
could be into the log, and could be done with a proper library. But just to
showcase the proper railguard that needs to be there.

## Part III - Deployment to Cloud

A first step for deployin to cloud, is to build a Dockerfile for our application
to live in. This has multiple pros, but the most importants are:
1. GCP has a method to directly deploy Docker containers.
2. It generates a specific environment for our application to work in, having
full control of what system's software resources are needed.

This dockerfile just installs a few system dependencies, and the application as
a module with poetry.

For the proper cloud deployment, a script was done to automate the process. As
it was suggested, the deployment was done on GCP.

The script was parametrized so that different names could be used for the cloud
artifacts, and make it more scalable if the deployment wanted to be made on the
deployment branch, or in the main branch.

This script uses the `gcloud` tool, provided by google, to manage GCP resources.
Orignally, the script expected to set the current project in this cli tool to
then deploy there, but for scalability reasons (having more than one project),
a specific project was selected.

The deployment script does:
1. Train the model with the available data.
2. Connects locally docker with gcloud.
3. Creates repo to push docker image generated and pushes it there.
4. Prompts gcloud with the running of the container.

This deploys the aplication in the 8080 port of the url:

`https://delay-model-api-qzo5moezqa-uw.a.run.app`

A modification was done to the makefile that generated the tests so that the
stress test would be done to this url.


## Part IV - CI/CD

For this part, as explicitly asked, copied the files from workflows to
.github/workflows, and completed with relevant instructions for github actions
to complete.

Then, after having an initial version of both, a rule was added to the
repository, such that the workflows need to be satisfactory to be able to merge
in the `develop`, `release` and `main` branches.

### CI
This workflow was defined to ensure the sanity of the code, by running the
functional tests each time the code is intended to be merged in the branches
mentioned above.

Notice that the model is trained in this stage. As the model performance test
is done during this stage, this would be the best place to test it, but only if
the training code has changed, to train it every time is a waste of time.
**This is a possible improvement to the actions done here**.

For the api is somewhat different. As it depends on the model, it might break
without changing anything from the `api.py` file. So, at least for the code we
have now, it should be tested everytime a `.py` file is modified.

The test should be done for both the model and the api in case the dependencies
are changed, to catch possible dependencies conflicts.

Ideally, if the model has been changed, it should be uploaded to a bucket so
that the continuous deployment stage can get it and upload it to the artifact
registry along with the docker container. This only need to happen if the merge
is into `main` or `develop` branches, as this is where the deployment need to be
tested.
**This is a possible improvement that can be done to the actions here defined**.
Notice that for this we should add github secrets.

### CD
This workflow is desing to automate the build, deploy and testing of the
application in the cloud provider of deployment.

For this, some configurations were done on the cloud provider (GCP) and some
secrets were added to github. This way, we have can access the cloud provider
securely from github actions.

Notice that in the first version, and as this is a challenge and not a true prod
server, some values were not put as secrets, such as the docker's *repo-name*,
docker's *container-name*, the applications *application_name* and the
*current_project* where to upload and deploy the application code. This
information is not necesarily sensitive, but might be in certain cases.
**A possible improvment in this case** would be to **Move the possibly sensitive
variables to github secrets**, and use those in the workflows, rather than the
variables defined in the bash script.

Also, instead of retraining the model in the bash script, take the model form a
cloud bucket as specified in the CI section.

More sensitive information, such as the GCP credentials, were always kept a
secret.

Also, another improvement would be to use to deployment repos, one for the
deployment stage, and one for the true production case when merging into main.

Also, the URL if deployment should be taken from the deployment script, and
passed onto the testing script, as this url is variable, and defined by the
cloud provider.
