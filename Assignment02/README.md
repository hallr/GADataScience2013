This R script performs n-fold cross validation for a kNN classifer.
This script can be run two ways:
   1. Perform a single run of n-fold cross-validation
   2. Run n-fold cross validation x times

So, for example, for 6-fold cross validation, option 1 will run
6-fold cross validation and return the n-fold generalization error
as averaged over those n-folds.

Option 2 enables you to run option 1 x times. It returns the average
n-fold generalization error for those x runs. In other words,
running option 2 with x=10 and nfolds=6 will peform 60 total
kNN classifications.
The code to run each option is included at the end of the script.
Be sure to comment out the line NOT being used.