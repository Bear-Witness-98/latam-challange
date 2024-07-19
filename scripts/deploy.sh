#!/bin/bash

# notice that to perform the deploy you should:
# * have a google could account
# * have a project setted up on google cloud
# * install and init google cloud cli
# * have your apis enabled in run.googleapis.com in your project (will be prompted to enable them)
#   * may cause an error the first time, as the request to run and enable the apis
#     generate a race condition

# settings to configure
repo_name="delay-model-image"
container_tag="delay-model-container"
service_name="delay-model-api"
current_project="santiago-tryolabs-latam"  # $(gcloud config get-value project)
region="us-west1"
DOCKER_IMAGE_FORMAT="docker.pkg.dev" # script is not intended to work by modifying this value
repo_address="${region}-${DOCKER_IMAGE_FORMAT}/${current_project}/${repo_name}"

# run training
python challenge/model.py

# allow docker to authenticate to gcp
gcloud auth configure-docker ${region}-${DOCKER_IMAGE_FORMAT}

# create repo to store docker image
gcloud artifacts repositories create \
    --repository-format=docker \
    --location=${region} \
    ${repo_name}

# build docker container with appropriate tag and push it
docker build . -t ${repo_address}/${container_tag} --platform linux/amd64
docker push ${repo_address}/${container_tag}

# deploy the image
gcloud run deploy ${service_name} \
    --image ${repo_address}/${container_tag} \
    --allow-unauthenticated \
    --region ${region}