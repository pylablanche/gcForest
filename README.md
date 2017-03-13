# gcForest 
Status : under development

This code contains a direct implentation of the gcForest algorithm as suggested in Zhou and Feng 2017.

The syntax uses the scikit learn style with a .fit() function to train the algorithm and a .predict() function to predict new values class.

So far it only works as a Classifier and does not include the following :

- Growth/evaluation split when training the Cascade Layers
- No parallelisation of the code has been done yet
