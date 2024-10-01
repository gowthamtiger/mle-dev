

# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## Project Structure

├── src
│   ├── __init__.py
│   ├── ingest_data.py
│   ├── train.py
│   └── score.py
├── tests
│   ├── __init__.py
│   ├── test_ingest.py
│   ├── test_train.py
│   └── test_score.py
├── README.md
└── env.yaml

## Install the required dependencies

## run python files and test it

## To excute the script
python < scriptname.py >

## Document it using Sphinx 




