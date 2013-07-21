# GA: Data Science - Homework 3 - Rob Hall
# n-fold cross-validation for Naive Bayesian classifer

fold.sizes <- function(nfolds, nrec, samesize=FALSE, echolevel=0)
{
    # calculate size of each fold, given number of folds and number of records in data
    # returns a vector of n fold sizes, in terms of number of records in each fold

    # nfolds: the desired number of folds
    # nrec: the number of records in the data set
    # echolevel: from zero to 3, controls amount written to screen
    # samesize: an optional boolean. Set to true to make each fold exactly the same
    #   size. This may result in the exclusion of a few records from the training
    #   and test sets. If samesize is false, then all the records will be used and
    #   the fold sizes may vary slightly.

    # For cases in which the number of records is evenly divisible by the number of folds,
    #   then the fold size is simply nrec/nfolds. However, for cases in which the
    #   number of records is not evenly divisible by the number of folds, then we
    #   must choose to either make all folds exactly the same size and discard a few records,
    #   or to make the folds as close to even as possible and use all the records.

    # NOTE: Use of samesize = TRUE is NOT recommended with the run.knn() function below,
    #   because the records "rounded off" of the folds will still be included in the
    #   test set. The test set is calculated as the whole set minus the test fold,
    #   not the union of all the other folds. See train.data <- data[-testindex, ] in run.knn().

    # initialize foldsizes as a vector
    foldsizes = c()

    if(samesize == FALSE)
        for (i in 1:nfolds) {
             first = 1 + (((i - 1) * nrec) %/% nfolds) #integer division
             last = ((i * nrec) %/% nfolds)
             foldsizes = append(foldsizes, last - first + 1)
        }
        else
        for (i in 1:nfolds) {          
             foldsizes = append(foldsizes, nrec %/% nfolds) #integer division
        }
    
    if(echolevel > 2) {
       cat('\n', 'number of folds = ', nfolds, '\n', sep='')
       cat('size of each fold:', foldsizes, '\n', sep=' ')
    }
    #print(foldsizes)
    return(foldsizes)
}

nfold.indices <- function(nfolds, nrec, samesize=FALSE, echolevel=0)
{
  # generate indices for n-fold cross validation
  # returns a list of vectors of fold indices
  #
  # nfolds: the number of folds for CV
  # nrec: the number of records in the data set
  # echolevel: from zero to 3, controls amount written to screen
 
  indices <- list()
  foldsizes <- fold.sizes(nfolds, nrec, samesize, echolevel)

  indexvalues <- 1:nrec
  for (i in 1:nfolds) {
    s <- sample(indexvalues, foldsizes[i]) # randomly sample enough index values to fill each fold
    indices[[i]] <- s  # add vector of index values for fold to index list
    indexvalues <- setdiff(indexvalues, s) # remove the index values sampled from the remaining pool
                                          # note setdiff function incredibly helpful here
  }
  # print(indices)
  return(indices)
}

nb.cv.error <- function(data, labelscol, nfolds, samesize=FALSE, echolevel=0)
{
  # returns n-fold generalization error for naiveBayes
  #
  # data: the data frame of records
  # labelscol: the numeric index of the column with the labels
  #        for the iris dataset, labelscol=5
  # nfolds: the number of folds for CV
  # echolevel: from zero to 3, controls amount written to screen

  labels <- data[ ,labelscol]
  data[ ,labelscol]<-NULL

  err.rates <- data.frame() # initialize results object

  # get our list of vectors of indices for each fold
  # foldindices will have as many elements as we have folds
  foldindices <- nfold.indices(nfolds, nrow(data), samesize, echolevel)

  for (i in 1:nfolds) {
    testindex <- foldindices[[i]] # get the vector of indices for this fold
   
    test.data <- data[testindex, ] # perform train/test split
    train.data <- data[-testindex, ] # note use of neg index...different than Python!
    test.labels <- as.factor(as.matrix(labels)[testindex]) # extract test set labels
    train.labels <- as.factor(as.matrix(labels)[-testindex]) # extract training set labels

    classifier <- naiveBayes(train.data, train.labels)
    nb.pred <- predict(classifier, test.data)

    if(echolevel > 2) {
       cat('\n', 'confusion matrix for fold number: ', i, '\n', sep='')
       print(table(nb.pred, test.labels, dnn=list('predicted','actual')))
    }

    this.err <- sum(test.labels != nb.pred) / length(test.labels)
    err.rates <- rbind(err.rates, this.err)
    
    if(echolevel > 1) cat('generalization error for fold', i, '=', this.err, '\n')
  }
  #print(err.rates)
  return(err.rates)
}

naiveBayes.nfold <- function(data, labelscol, nfolds, samesize=FALSE, echolevel=0)   
{
  # returns n-fold generalization error for Naive Bayesian classifer
  #
  # data: the data frame of records
  # labelscol: the numeric index of the column with the labels
  #        for the iris dataset, labelscol=5
  # nfolds: the number of folds for CV
  # echolevel: from zero to 3, controls amount written to screen

   stopifnot(nfolds>1) #minimum of 2 folds for result to make sense
  
   err.rates <- nb.cv.error(data, labelscol, nfolds, samesize, echolevel)
   nfold.generr <- sum(err.rates) / nfolds

   if(echolevel > 0) cat('\n', 'n-fold generalization error across ', nfolds, ' folds = ', nfold.generr, '\n', sep='')

   return(nfold.generr)
}

x.by.nfold <- function(data, labelscol, nfolds, x=10, samesize=FALSE, echolevel=0)
{
   # runs n-fold cross-validation x times and averages the x averages
   #    to obtain a better estimate of generalization error
   #
   # x: the number of times to run through n-fold cross validation
   # echolevel: from zero to 3, controls amount written to screen

   stopifnot(nfolds>1) #minimum of 2 folds for result to make sense

   nfold.err.rates <- data.frame() # initialize results object

   for (i in 1:x) {
      set.seed(1+(i * 100)) #varying seed for each run of n-fold cv

      nfold.generr <- naiveBayes.nfold(data, labelscol, nfolds, samesize, echolevel)
      nfold.err.rates <- rbind(nfold.err.rates, nfold.generr)
   }
   xby.nfold.generr <- sum(nfold.err.rates) / x

   if(echolevel >0) cat('\n', 'average of ', x, ' n-fold generalization errors = ', xby.nfold.generr, '\n', sep='')

   return(xby.nfold.generr)
}

library("e1071")
#library(ggplot2)

set.seed(1234) #seed is reset by x.by.nfold(), if called

# Code below runs a single n-fold cross validation.
generr <- naiveBayes.nfold(data=iris, labelscol=5, nfolds=6, samesize=FALSE, echolevel=3)

# If running the code below, then comment out the single-run n-fold line of code above.
# Code below runs the n-fold cross-validation x times and averages the x averages.
#xby.err <- x.by.nfold(data=iris, labelscol=5, nfolds=6, x=10, samesize=FALSE, echolevel=2)