# gcForest 
Status : under development

License : MIT

gcForest is an algorithm suggested in Zhou and Feng 2017.
It uses a multi-grain scanning approach for data slicing and a cascade structure of random forests layers (see paper for details).

The present code contains a first direct python (3.x) implentation of the gcForest algorithm. The syntax uses the scikit learn style with a .fit() function to train the algorithm and a .predict() function to predict new values class. You can find two examples in the jupyter notebook included in the repository.

**gcForest** has been developed as a Classifier and designed such that :
- It will be possible to use multi grain scanning and cascade forest separately (needs to be tested)
- Future parallelization (CPU/GPU) will be easily implemented

Note that I wrote the code from scratch in two days and even though I have tested it on several cases I cannot certify that it is a 100% bug free.
**Feel free to test it and send me your feedback about any improvement and/or modification!**


--- Some Tests Results using this code ---
(will be updated as result come out)

Scikit-learn handwritten digits classification :

training time ~ 5min

accuracy ~ 98%
