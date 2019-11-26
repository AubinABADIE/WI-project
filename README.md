# WI-project - Cortex Group

The purpose of this project is to provide a click on ad prediction according to the information about users (RTB).
The Cortex system takes a json file containing users as input data and produces a csv file containing the forecast result from a sample of the initial data.

There are three models that can be used:

1) Logistic Regression
2) Naive Bayes
3) Random Forest

To reach the forecasting we have the following steps:
- Clean the input data.
- Build the model with training data (80%).
- Test the model with test data (20%).
- Write the result in a csv file in data/prediction/ folder.


## How to use

### Your input json file must be inside the data folder

Start sbt shell using
* sbt

Launch the program using
* run nameOfJsonFile model

Where model can be:
* _naive_ for Naive Bayes
* _lr_ for Logistic Regression
* _rf_ for Random Forest 

## Requirements

Required: sbt 0.1 and scala must be installed.