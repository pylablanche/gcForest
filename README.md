# gcForest

*Status* : under development

**gcForest** is an algorithm suggested in Zhou and Feng 2017 ( https://arxiv.org/abs/1702.08835 ). It uses a multi-grain scanning approach for data slicing and a cascade structure of multiple random forests layers (see paper for details).

**gcForest** has been first developed as a Classifier and designed such that the multi-grain scanning module and the cascade structure can be used separately. During development I've paid special attention to write the code in the way that future parallelization should be pretty straightforward to implement.

## Prerequisites

The present code has been developed under python3.x. You will need to have the following installed on your computer to make it work :

* Python 3.x
* Numpy >= 1.12.0
* Scikit-learn >= 0.18.1
* jupyter >= 1.0.0 (only useful to run the tuto notebook)

You can install all of them using `pip` install :

```sh
$ pip3 install requirements.txt
```

## Using gcForest

The syntax uses the scikit learn style with a `.fit()` function to train the algorithm and a `.predict()` function to predict new values class. You can find two examples in the jupyter notebook included in the repository.

```python
from GCForest import *
gcf = gcForest( **kwargs )
gcf.fit(X_train, y_train)
gcf.predict(X_test)
```


## Notes
I wrote the code from scratch in two days and even though I have tested it on several cases I cannot certify that it is a 100% bug free obviously.
**Feel free to test it and send me your feedback about any improvement and/or modification!**

## Built With

* [PyCharm](https://www.jetbrains.com/pycharm/) community edition


## License
This project is licensed under the MIT License (see `LICENSE` for details) 



### Early Results 
(will be updated as new results come out)

* Scikit-learn handwritten digits classification :<br>
training time ~ 5min <br>
accuracy ~ 98%
