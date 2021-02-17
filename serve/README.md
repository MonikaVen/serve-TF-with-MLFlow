# Vinted HW

## Instructions

In order for application to run, you will need docker-compose and docker installed:

```
sudo snap install docker
```

https://docs.docker.com/compose/install/

Download the code from repository.
When inside a Vinted_HW folder, run:

```
sudo docker-compose up --build
```

You will see two dockers being built (these are two serving apps) and one docker being downloaded (tensorflow/serving).

As soon as the process is finished, you will be able to visit these two sites in order to test the models:

```
http://localhost:5000/ 
```
for Flask version of an app.

and

```
http://localhost:8000/
```
for FastAPI version of an app.

 On both sites you will be able to submit an image with a click on "Browse" and picking a file.
JPEG, JPG and PNG are supported.

Then you will click "Upload" or "Submit Query" and image will be uploaded, processed and sent to the model.

Model is hosted with the tensorflow/serving docker.

Both apps measure the time needed to make a prediction (as one of the key KPI's for ML) and outputs the label of most likely predictions.

Everything is displayed back to the user.

## Potential improvements

Given the real-world scenario containers could be run on Kubernetes for scalability and redundancy.
However, they are quite resource intensive and require more dependencies and complexity when setting up, 
which would not be a good choice for a project of this format.

ML accuracy could be measured and improved. It is as well one of the main KPI's to track for ML.
Parameters within model could be optimized better.

Both of the KPI's and container availability could be logged and displayed with Prometeus and Grafana
for simple MLOps monitoring.

Data drift can also be tracked with the average values of inflowing data.

Http protocol could be encrypted, so the app would use https for better data security.
