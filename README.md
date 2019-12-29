### Usage
- install required packages with pipenv command --dev
```
pipenv install --dev
```
- run main function with pipenv command run start
```
pipenv run start
```

## Installation of TPOT

You will find [installation][install] in the official site.
- upgrade pip to the latest version
```
pip install upgrade pip
```
- install required packages
```
pip install numpy scipy scikit-learn pandas joblib
```
- install tpot
```
pip install tpot
```

## auto-sklearn example

You will find [examples][example] in the official site.
```
from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data.astype(np.float64),
    iris.target.astype(np.float64), train_size=0.75, test_size=0.25, random_state=42)

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_iris_pipeline.py')
```

[install]: https://epistasislab.github.io/tpot/installing/ "TPOT official"
[example]: https://epistasislab.github.io/tpot/examples/ "TPOT examples"

