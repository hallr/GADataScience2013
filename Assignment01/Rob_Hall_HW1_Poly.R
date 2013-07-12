# GA: Data Science - Homework 1 - Rob Hall
# Polynomial regression script

# This script intended to be run with the command:
#   source("Rob_Hall_HW1_Poly.R", echo=TRUE, skip.echo=30, max.deparse.length = 500)

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

# Load dataset and set column header names
# Home page of dataset is: http://archive.ics.uci.edu/ml/datasets/Housing
# Set column header names load dataset
# Data set does not have header names, so I add those manually. Header names match the data.names file

library(ggplot2)

cnames <- c("CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV")
housing <- read.table('http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', col.names = cnames, sep="", h=FALSE)


plot(housing)  # It appears that some features may be highly correlated (LSTAT and RM)
cor(housing)  # It appears that LSTAT and MEDV are negatively correlated (-0.7377), which is intuitive because
              # lower income people can only afford lower priced homes

# For this exercise, the instructions are to run a regression on a single factor that is significant
# From the previous exercise in multiple regression, we see that that factor RM (the size of the house)
# is a very significant predictor of the house price (MEDV).

# Try fitting a second order polynomial
fit.poly2 <- lm(MEDV ~ poly(RM, degree = 2), data=housing)
summary(fit.poly2)

# Try fitting a third order polynomial
fit.poly3 <- lm(MEDV ~ poly(RM, degree = 3), data=housing)
summary(fit.poly3)

# Try fitting a fourth order polynomial
fit.poly4 <- lm(MEDV ~ poly(RM, degree = 4), data=housing)
summary(fit.poly4)

# Try fitting a fifth order polynomial
fit.poly5 <- lm(MEDV ~ poly(RM, degree = 5), data=housing)
summary(fit.poly5)

# Fourth and fith order polynomial terms have significant t-values, and values of
#  R^2 and adjusted R^2 continue to increase slightly...

# Let's use a 3rd order polynomial for the overfitting exercise.

# Now, let's add more variables to demonstrate overfitting
# First, we'll add LSTAT, because it has the next most signfic
fit.poly3.2f <- lm(MEDV ~ poly(RM, degree = 3) + poly(LSTAT, degree = 3), data=housing)
summary(fit.poly3.2f)

# Interesting - the t-value for the 3rd order polynomial term on RM is no longer signficant.

# Now, we will add another variable. This time, the commute distance DIS (next most significant t-value in fit3)
fit.poly3.3f <- lm(MEDV ~ poly(RM, degree = 3) + poly(LSTAT, degree = 3) + poly(DIS, degree = 3), data=housing)
summary(fit.poly3.3f)

# Now, we will add another variable. This time, the pupil-teacher ratio.
fit.poly3.4f <- lm(MEDV ~ poly(RM, degree = 3) + poly(LSTAT, degree = 3) + poly(DIS, degree = 3) + poly(PTRATIO, degree = 3), data=housing)
summary(fit.poly3.4f)

# Now, we will add another variable. This time, air pollution.
fit.poly3.5f <- lm(MEDV ~ poly(RM, degree = 3) + poly(LSTAT, degree = 3) + poly(DIS, degree = 3) + poly(PTRATIO, degree = 3) + poly(NOX, degree = 3), data=housing)
summary(fit.poly3.5f)

# As expected, R^2 continues to increase as we add more explanatory variables (adding explanatory variables won't decrease it).
# However, the marginal increase in R^2 rapidly decreases.


plot(resid(fit.poly3.5f)) # want to see absence of structure in resid scatterplot ("gaussian white noise")
qqnorm(resid(fit.poly3.5f))
# want to see straight diagonal line in resid qqplot
# Not the straight diagonal line we like to see.

