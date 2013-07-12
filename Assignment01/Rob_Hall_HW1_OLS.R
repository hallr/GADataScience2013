# GA: Data Science - Homework 1 - Rob Hall
# Linear regression script

# This script intended to be run with the command:
#   source("Rob_Hall_HW1_OLS.R", echo=TRUE, skip.echo=30, max.deparse.length = 500)

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

print(plot(housing)) # It appears that some features may be highly correlated (LSTAT and RM)
print(cor(housing))
# It appears that LSTAT and MEDV are negatively correlated (-0.7377), which is intuitive because
# lower income people can only afford lower priced homes

# As part of exploring the data, let's plot each of the variables against home value.
max.colindex <- 13

for (i in 1:max.colindex)
{
  plot <- ggplot(housing, aes(x = housing[,i], y = MEDV)) + geom_point() + xlab(names(housing[i]))
  print(plot)
}
# So, there is clearly a linear relationship to RM
#   Also, there is a non-linear relationship to LSTAT

# We saw in the correlations that RM and LSTAT are corelated. Let's plot that relationship.
plot.rm.lstat <- ggplot(housing, aes(x = LSTAT, y = RM)) + geom_point()
print(plot.rm.lstat)
# Yes, there is some collinearity here between RM and LSTAT, which is intuitive.

# All the features are continuous except for whether the property is on the Charles River.
# Coerce CHAS in housing and test as a factor
housing$CHAS <- as.factor(housing$CHAS)

# MEDV is the median value of owner-occupied homes in $1000's. For this model, we will regress it on
#   the full set of variables.
fit <- lm(MEDV ~., data=housing)
summary(fit)

# Forcing the y-intercept through the origin creates an artificially high R^2. So, we will not do that here.
# fit <- lm(MEDV ~ 0+., data=housing)
# summary(fit)

# Remove the AGE factor because its t-value is very small (= 0.052).
fit2 <- update(fit, .~. -AGE)
summary(fit2)

# Remove the INDUS factor because its t-value is now the smallest (= 0.335).
fit3 <- update(fit2, .~. -INDUS)
summary(fit3)

#fit3 appears to be the optimal OLS model, because all remaining features have |t| > 1.0

# To explore further, remove the CHAS factor because its t-value is now the smallest (= 3.183).
# CHAS is a Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
fit4 <- update(fit3, .~. -CHAS)
summary(fit4)
# R^2 now went down, presumably because we removed a factor with significant explanatory power

# So, fit3 appears to be the optimal OLS model. fit3 also minimizes RSE.

# Out of curiosity, let's try removing one more factor
# Remove the ZN factor because its t-value is now the smallest (= 3.352).
fit5 <- update(fit4, .~. -ZN)
summary(fit5)
# Indeed, R^2 continues to decline.

plot(resid(fit3)) # want to see absence of structure in resid scatterplot ("gaussian white noise")
# Does not appear entirely random. Notable vertical "line" at about 360 on the x-axis

qqnorm(resid(fit3)) # want to see straight diagonal line in resid qqplot
# Hmmm, not quite a straight diagonal line, although not awful.

###############################################
# New linear model, with some transformations
###############################################

# Should we transform LSTAT to ln(LSTAT), because of the non-linear relationship?
plot.log.lstat <- ggplot(housing, aes(x = log(LSTAT), y = RM)) + geom_point()
print(plot.log.lstat)
# The relationship looks much more linear after the transformation!

# New model, with log(LSTAT) instead of LSTAT.
fit.transform1 <- lm(MEDV ~ log(LSTAT) + CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B, data=housing)
summary(fit.transform1)
# Both R^2 and adjusted R^2 increase!

# With additional time, try scaling the variables
