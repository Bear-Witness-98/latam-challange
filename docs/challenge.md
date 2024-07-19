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
