---
title: "Weight_Lifting_Prediction"
author: "Jamazi"
date: "2/12/2019"
output:
  html_document:
    keep_md: yes
  pdf_document: default
---



## Summary

We built a model to predict whether someone was doing a barbell lift correctly based on data from a gyroscopic fitness device. We used training data which had five categories of barbell lifting corresponding to correct and incorrect form. First we read in the training data and remove variables with a high number of NA values:



```r
wd <- paste(getwd(),"/pml-training.csv", sep="")
pml <- read.csv(file=wd, header=TRUE, na.strings=c("", "NA"))

remove <- 1:7
for (i in 1:ncol(pml)) {
    if (sum(is.na(pml[,i]))>5000 )  {
        remove <- c(remove, i)
    }
}
pml <- pml[,-remove]
```

Next we remove variables ending in "x", "y", and "z" because they are likely summarized by other variables. Then we separate our training data into "training", "testing", and "validation" sets.


```r
remove2 <- grep(pattern = "x|y|z", x=names(pml))
pml <- pml[,-remove2]

set.seed(1234)
inBuild <- createDataPartition(y=pml$classe, p=0.7, list=FALSE)

validation <- pml[-inBuild,]
buildData <- pml[inBuild,]

inTrain <- createDataPartition(y=buildData$classe, p=0.7, list=FALSE)
training <- buildData[inTrain,]
testing <- buildData[-inTrain,]

dim(training); dim(testing); dim(validation)
```

```
## [1] 9619   13
```

```
## [1] 4118   13
```

```
## [1] 5885   13
```

We see that each of our data sets has thousands of entries, so they should be large enough to build and test robust training models. This method of separating our training data into sections against which we can test our prediction models is called cross-validation.

## Creating Prediction Models

We're going to create three different prediction models and combine them to increase the accuracy of our predictions. We use the random forrest (rf), gradient boosting machine (gbm), and linear discriminant analysis and combine the results to extract the benefits of each. 


```r
#use decision tree model as a predictor
library(caret)
modFit <- train(classe ~., method="rf", data=training, preProcess="pca")
rfPred <- predict(modFit, testing)
#results in ~90% accuracy

#use boosting model as a predictor
library(ISLR)
modFit2 <- train(classe~., method="gbm", data=training, verbose=FALSE)
bPred <- predict(modFit2, testing)
#results in ~90% accuracy

#use lda model as a predictor
modFit3 <- train(classe~., method="lda", data=training)
LDApred <- predict(modFit3, testing)
#results in ~43% accuracy
```

Our three models predicted the type of exercise of our testing subset with 90%, 90%, and 43% accuracy respectively. Using just the first or second method would be okay, but we'll try to do better by stacking these models, training them on our testing subset, and testing them on our validation subset.


```r
#combine predictors and train based on test set
predDF <- data.frame(rfPred, bPred, LDApred, classe=testing$classe)
combModFit <- train(classe~., method="rf", data=predDF)
combPred <- predict(combModFit, predDF)
#results in 94.4% accuracy

#predict on validation set
pred1V <- predict(modFit, validation)
pred2V <- predict(modFit2, validation)
pred3V <- predict(modFit3, validation)
predVDF <- data.frame(rfPred=pred1V, bPred=pred2V, LDApred=pred3V)
combPredV <- predict(combModFit, predVDF)
confusionMatrix(combPredV, validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1615   35   15    9    3
##          B   40 1026   23   24    9
##          C    2   50  939   23    8
##          D    9    7   40  899   26
##          E    8   21    9    9 1036
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9371          
##                  95% CI : (0.9306, 0.9432)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9205          
##  Mcnemar's Test P-Value : 3.216e-07       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9648   0.9008   0.9152   0.9326   0.9575
## Specificity            0.9853   0.9798   0.9829   0.9833   0.9902
## Pos Pred Value         0.9630   0.9144   0.9188   0.9164   0.9566
## Neg Pred Value         0.9860   0.9763   0.9821   0.9867   0.9904
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2744   0.1743   0.1596   0.1528   0.1760
## Detection Prevalence   0.2850   0.1907   0.1737   0.1667   0.1840
## Balanced Accuracy      0.9750   0.9403   0.9491   0.9580   0.9739
```

```r
#results in 93.7% accuracy
```

We combined our prediction models and trained a random forrest model based on their combined predictions to build a combined model. This combined model was able to predict the type of exercise in our validation subset with 93.7% accuracy. 
