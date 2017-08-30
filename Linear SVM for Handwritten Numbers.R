##Kenny Warner

## From Statistical Machine Learning Class at Columbia University
## Goal is to apply a support vector machine to classify
## handwritten digits 
## In this I use the e1071 library for the implementation of the SVM

##Each row of the datasets used represent a vector of length 256
##This vector represents a 16 x 16 handwritten number
## There are two classes 5 and 6, labeled as -1 and +1

library(e1071)
base_loc = "C:\user\kenny\documents\intro data science"

digit5 = read.table(paste(base_loc, "train.5.txt", sep=""), header=F, sep=",")
digit6 = read.table(paste(base_loc, "train.6.txt", sep=""), header=F, sep=",")

n1 = dim(digit5)[1]
n2 = dim(digit6)[1]
p =  dim(digit5)[2]

## creating training and testing set 
test_prop = 0.2
index1 = sample(n1, round(test_prop * n1), replace=F)  # index for testing
index2 = sample(n2, round(test_prop * n2), replace=F)

xtest  = rbind(digit5[index1, ], digit6[index2, ])
xtrain = rbind(digit5[-index1, ], digit6[-index2, ])
ytest  = factor(c(rep("5", length(index1)), rep("6", length(index2))))
ytrain = factor(c(rep("5", n1 - length(index1)), rep("6", n2 - length(index2))))

## Use linear SVM with k = 10 fold CV
# default for tune.svm() is 10-fold CV
tune_svm1 = tune.svm(x = xtrain, y = ytrain,
                     cost = seq(0.001, 0.08, 0.002), kernel="linear")
pdf(width = 7, height = 7, file = paste(base_loc, "3-1.pdf", sep=""))
plot(tune_svm1, main = "tuning on Cost(C) of linear-SVM")
dev.off()

tune_svm2 = tune.svm(x = xtrain, y = ytrain, cost = seq(1, 10, 2),
                      gamma = seq(0.004, 0.04, 0.004), kernel="radial")
pdf(width = 7, height = 7, file = paste(base_loc, "3-2.pdf", sep=""))
plot(tune_svm2, type = "contour", main = "tuning on Cost(C) and Gamma of RBF-SVM")
dev.off()

## Train the best model and then compare the results on the test set
bestsvm1 = tune_svm1$best.model
bestsvm2 = tune_svm2$best.model

pred_svm1 = predict(bestsvm1, xtest)
pred_svm2 = predict(bestsvm2, xtest)

err1 = sum(pred_svm1 != ytest) / length(ytest)
err2 = sum(pred_svm2 != ytest) / length(ytest)


