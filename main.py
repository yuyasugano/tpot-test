#!/usr/bin/python
from tpot import TPOTClassifier
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import time

def tpot_classifier(X, y):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=39)
    pipeline_optimizer = TPOTClassifier(generations=10, population_size=10, cv=5, random_state=42, verbosity=2)
    pipeline_optimizer.fit(X_train, y_train)
    pipeline_optimizer.export('tpot_pipeline.py')
    print("Accuracy score", pipeline_optimizer.score(X_test, y_test))

    return pipeline_optimizer

def main():
    X, y = sklearn.datasets.load_iris(return_X_y=True)
    ret = tpot_classifier(X, y)
    print(ret)

if __name__ == '__main__':
    print("Started...")
    start = time.time()
    main()
    elapsed_time = time.time() - start
    print("Elapsed time: {}".format(elapsed_time) + "[sec]")

