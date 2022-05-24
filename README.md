## Decision Tree Implementation

This is an implementation of a Decision Tree using Python and Numpy library. Its goal is to tell whether a person has a hearth disease or not. The created model inherits from sklearn BaseEstimator and so it accepts multiple sklearn modules like cross-validate.
It is created with the aim to resemble models from machine learning libraries. Therefore, it can be used as one would expect by first creating the model, followed by calling fit and predict functions.

This Decision Tree features:
- Multiple heurististics and coefficient to choose
- Fixing outliers, null values, categorial encoding, etc...
- Probabilistic Approach
- 2-way partitioning
- Random Forest
- Own k-Fold cross validation function

### How to run
Install dependencies
```bash
python3 -m pip install -r requirements.txt
```

Run program
```bash
python ./src/main.py
```
By default, the program runs a batch of different tests to check the performance using multiple options. The file main.py can be easily modified and comment the tests that are not to be run.
