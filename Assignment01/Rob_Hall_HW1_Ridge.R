# GA: Data Science - Homework 1 - Rob Hall
# Ridge regression script

# This script intended to be run with the command:
#   source("Rob_Hall_HW1_Ridge.R", echo=TRUE, skip.echo=34, max.deparse.length = 500)

# Feature definitions
#   1. CRIM      per capita crime rate by town
#   2. ZN        proportion of residential land zoned for lots over 
#                25,000 sq.ft.
#   3. INDUS     proportion of non-retail business acres per town
#   4. CHAS      Charles River dummy variable (= 1 if tract bounds 
#                river; 0 otherwise)
#   5. NOX       nitric oxides concentration (parts per 10 million)
#   6. RM        average number of rooms per dwelling
#   7. AGE       proportion of owner-occupied units built prior to 1940
#   8. DIS       weighted distances to five Boston employment centres
#   9. RAD       index of accessibility to radial highways
#   10. TAX      full-value property-tax rate per $10,000
#   11. PTRATIO  pupil-teacher ratio by town
#   12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
#                by town
#   13. LSTAT    % lower status of the population
#   14. MEDV     Median value of owner-occupied homes in $1000's

# The lm.ridge function is part ofthe MASS package. So, we must load that.
library(MASS)
# Help for lm.ridge can be obtained via ?MASS::lm.ridge
# Help for select() can be obtained via ?MASS::select

# As it turns out, the MASS package has an data object called housing. So, I will use a different name for our data.
# Home page of dataset is: http://archive.ics.uci.edu/ml/datasets/Housing
# Set column header names load dataset
# Data set does not have header names, so I add those manually. Header names match the data.names file

cnames <- c("CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV")
houseprices <- read.table('http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', col.names = cnames, sep="", h=FALSE)

# For this exercise, we will use ridge regression and then compare some of the coefficients and
#    predictions with those obtained from OLS regression

# First, let's scale the continuous variables. Note that we must convert the output of the scale() function
#   back into a data frame in order for lm.ridge to work
houseprices.scaled <- as.data.frame(scale(houseprices))

# Use the select() function to search for lambda
select(lm.ridge(MEDV ~ ., data=houseprices.scaled, lambda = seq(0,1,0.001)))
# Smallest value of GCV is at 1, so let's expand the search range
select(lm.ridge(MEDV ~ ., data=houseprices.scaled, lambda = seq(0,10,0.01)))
# Smallest value of GCV is now at 4.26, so our search range for lambda seems good

houseprices.ridge <- lm.ridge(MEDV ~ ., data=houseprices.scaled, lambda = 4.26)

# Print ridge regression coefficients
houseprices.ridge 

# Let's round those coefficients to 3 decimal places for easier reading
round(houseprices.ridge$coef, 3)

# Next, compare the coefficients with those from OLS regression, which is same as lamba = 0
houseprices.ols <- lm.ridge(MEDV ~ ., data=houseprices.scaled, lambda = 0)
round(houseprices.ols$coef, 3)

# As expected, the coefficients for the the linear model and the ridge regression are very close,
#    but not exactly equal.

# I'm also curious to make sure that the linear model from the lm() function is the same
#     as the ridge model from lm.ridge with lambda set to zero.
houseprices.lm <- lm(MEDV ~ ., data=houseprices.scaled)
round(houseprices.lm$coef, 3)
# Yes, indeed it is.

# The only thing left is to compare some predictions
# To make the code a little neater, let's define an array for our new data point first.
newdata <- as.data.frame(c(CRIM=-0.42, ZN=0.28, INDUS=-1.29, CHAS=-0.27, NOX=-0.14, RM=0.41, AGE=-0.12, DIS=0.14, RAD=-0.98, TAX=-0.67, PTRATIO=-1.46, B=0.44, LSTAT=-1.10))
# Next time, it will be better to define training data and test data, instead of hard coding a test data point.

predict.ols <- predict.lm(houseprices.lm, newdata)
summary(predict.ols)
# Hmmm, apparently there is no predict() function that works for objects of class "ridgelm". So, we'll implement it.
predict.ridge <- houseprices.scaled %*% houseprices.ridge$coef + houseprices.ridge$ym
summary(predict.ridge)
