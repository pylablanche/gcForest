# gcForest in Python

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

### Known Issues

**Memory comsuption when slicing data**
There is now a short naive calculation illustrating the issue in the notebook.
So far the input data slicing is done all in a single step to train the Random Forest for the Multi-Grain Scanning. The problem is that it might requires a lot of memory depending on the size of the data set and the number of slices asked resulting in memory crashes (at least on my Intel Core 2 Duo).<br>
*I have recently improved the memory usage (from version 0.1.4) when slicing the data but will keep looking at ways to optimize the code.*

**OOB score error**
During the Random Forests training the Out-Of-Bag (OOB) technique is used for the prediction probabilities. It was found that this technique can sometimes raises an error when one or several samples is/are used for all trees training.<br>
*A potential solution consists in using cross validation instead of OOB score although it slows down the training. Anyway, simply increasing the number of trees and re-running the training (and crossing fingers) is often enough.*

## Built With

* [PyCharm](https://www.jetbrains.com/pycharm/) community edition
* ``memory_profiler`` library

## License
This project is licensed under the MIT License (see `LICENSE` for details) 



### Early Results 
(will be updated as new results come out)

* Scikit-learn handwritten digits classification :<br>
training time ~ 5min <br>
accuracy ~ 98%
