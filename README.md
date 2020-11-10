# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The data for use in this project is the UCI Bank Marketing dataset on client responses acquired through a direct call marketing campaign by a Portuguese banking Institution with the aim to access whether a client would subscribe for the bank term deposit given as a ‘yes’, or a ‘no’. Therefore, the problem suggests a binary classification problem where the goal is to predict if the client will subscribe a term deposit. The data can be found here: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

For this classification task of predicting, ‘yes’ or ‘no’, the Voting Ensemble model from the AutoML pipeline emerged as the best forming model by a 91.68% accuracy. 

# The Azure ML Pipelines

## Scikit-learn Pipeline
In this project, the Azure ML SDK services is employed to train a standard Scikit-learn Logistic Regression model on the Bank Marketing ‘tabular’ dataset on a single-node CPU through employing the capabilities of the Azure ML HyperDrive for optimizing the hyperparameters. The pipeline is illustrated below.

![]

Considering that the HyperDrive requires a search space for tuning the hyperparameters, this project adopted the Random Sampling search space with the intention to randomly sample hyperparameter values from a defined search space without incurring computational cost. This search space supports hyperparameter values provided as discrete or continuous values; but this project search space is set to discrete for both the Regularization parameter, C, and the Maximum number of iterations, Max-iter, because it achieved the best accuracies compared to the accuracies obtained of the model when the continuous search space was used.
Further, the Random search space supports early termination of low-performing models. To apply the early stopping policy, this project adopted the “Bandit Termination Policy” to ensure that the Azure ML pipeline does not waste time exploring runs with hyperparameters that are not promising. The policy is expressed as:

T=  Metric/((1+S) )                           where T is the termination threshold,S,the slack-factor. 

A run terminates when metric < T.


## AutoML Pipeline
An AutoML is built on the Bank Marketing dataset to automatically train and tune machine learning algorithms at various hyperparameter tuning and feature selection for an optimal selection of a model that best fits the training dataset using a given target metric. The pipeline is diagrammatically provided below.

![]
![]

## Pipeline comparison
From the experimental results of the Azure Machine Learning pipelines (Scikit-Learn and AutoML): the AutoML pipeline is observed to produce the best performing model, Voting Ensemble, which showed to be a better fit to the data by its 91.68% accuracy. Though, the accuracy achieved with Voting Ensemble is a marginal difference of 0.67% compared to the 91.01% accuracy achieved with Scikit-Learn hyperparameter tuned Logistic Regression model, the performance difference can be attributed to the weighting mechanism that AutoML automatically applies to imbalanced data. The Scikit-Learn Logistic Regression model without hyperparameter tuning is by far the lowest performing model by its 90.97% accuracy achieved with values of 1 and 100, that is, Regularization strength, C, and Maximum Iteration, max-iter, parameters, respectively. 

## Future work
On visualizing the data by class using the Azure ML feature importance service, it is obvious that there is an imbalance in the data class. The ‘yes’ class have more data than the ‘no’ class. 

![]


Therefore, there are numerous performance improvement strategies that can be explored. They are:
    •	 A better metric such as AUC metric or the F1 metric can be optimized because they are insensitive to class imbalance. This applies to both the Scikit-Learn pipeline           and AutoML. 
    •	Class balancing techniques such as up-sampling the smaller class, ‘no’, or down-sampling the larger class, ‘yes’; these methods can help to prevent the Scikit-Learn            Logistic Regression Model from overfitting. 
    •	For the Scikit-Learn Logistic Regression Model, the kernel selection can be optimized for the given data. 
    •	For the AutoML, the cross-validation hyperparameter can be experimented to find the best cross-validation fold for the given data.

