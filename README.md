# Gender Classification from Text
This project is a machine learning-based solution for gender classification from textual data. 
The goal of the project is to predict the gender of an individual based on the text they have written.

## Data
The dataset used for training and testing the models is a collection of short stories written by various authors. 
The dataset is split into two parts, a training set and a test set. 
The training set is used to train the models, and the test set is used to evaluate the performance of the models.

## Preprocessing
The textual data is preprocessed by cleaning the text and removing any unwanted characters, such as punctuation.
The text is then transformed into numerical features using the TF-IDF vectorizer. 
Feature selection is performed using mutual information, and the data is scaled using MinMaxScaler.

## Models
Three different classification models are used for gender classification: 
- Logistic Regression: lr 
- SGDClassifier: Stochastic Gradient Descent Classifier
- Perceptron: Perceptron Classifier
- MLPClassifier: Multi-Layer Perceptron Classifier
- LinearSVC: Linear Support Vector Classification
- SVC: Support Vector Classification
- DecisionTreeClassifier: Decision Tree Classifier
- KNeighborsClassifier: K-Nearest Neighbors Classifier

The models are trained and evaluated using 10-fold cross-validation.

## Ensemble Model
An ensemble model is created by stacking the models. 
The output of the models is used as input to a meta-classifier, which is trained on the stacked features. 
The ensemble model is evaluated using 10-fold cross-validation.

## Results
The performance of the individual models and the ensemble model is evaluated using various metrics, including accuracy, precision, recall, and F1-score. 
The results show that the ensemble model outperforms the individual models in terms of F1-score, with an F1-score of 0.716.

Dependencies
The project requires the following dependencies:

- Python 3.7+
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
