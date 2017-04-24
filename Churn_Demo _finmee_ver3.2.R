rm(list = ls())
setwd("D:/Work/fwdchurnpredictionfiles")
library(readr)
churnData <- read.csv("D:/Work/fwdchurnpredictionfiles/credit card churn-data-both - Copy.csv")
row<- nrow(churnData)
col <- ncol(churnData)
cat("File is imported...\n No. of records=",row,"\n No.of Columns=",col)

#churnData Exploration
str(churnData)
summary(churnData)
head(churnData)
tail(churnData)
sum(is.na(churnData))

churnData$TARGET <- as.factor(churnData$TARGET)
churnData$N_EDUC <- as.factor(churnData$N_EDUC)
churnData$SX<-as.factor(churnData$SX)
churnData$CIV_ST<-as.factor(churnData$CIV_ST)

#about original target varaiable
table(churnData$TARGET)
print(prop.table(table(churnData$TARGET)))

#Feature Selection
library(MASS)
md= glm(churnData$TARGET~., churnData, family=binomial(link = "logit"))
x<-stepAIC(md,direction="both")
selectedcols<-attr(terms(x),"term.labels")
cat("After feature selection, No.of important features=",length(selectedcols))
print(selectedcols)
write(selectedcols, "important_features.txt")
newchurnData<-churnData[selectedcols]
newchurnData$TARGET<-churnData$TARGET

#Data Partitioning
library(caret)
set.seed(1234)
splitIndex <- createDataPartition(newchurnData$TARGET, p = .70,
                                  list = FALSE,
                                  times = 1)
trainchurnData <- newchurnData[ splitIndex,]
sum(is.na(trainchurnData))
testchurnData <- newchurnData[-splitIndex,]
dim(trainchurnData)
dim(testchurnData)
prop.table(table(trainchurnData$TARGET))
prop.table(table(testchurnData$TARGET))

#SMOTE for data balancing
library(DMwR)
trainchurnData <- SMOTE(TARGET~.,trainchurnData,perc.over = 443,perc.under = 233)
prop.table(table(trainchurnData$TARGET))
table(trainchurnData$TARGET)
dim(trainchurnData)

# Model-1: C5.0 Tree
library("C50")
library(partykit)
######Training Phase##########
c5.0_moldel= C5.0(trainchurnData$TARGET~.,data = trainchurnData)
write(capture.output(summary(c5.0_moldel)), "churnc5.0ModelTrainSummary.txt")
modparty<-as.party(c5.0_moldel)
rls <- partykit:::.list.rules.party(modparty)
rval <- data.frame(response = predict(modparty, type = "response"))
rval$prob <- predict(modparty, type = "prob")
rval$rule <- rls[as.character(predict(modparty,type="node"))]
write.csv(rval,"churnallrulestrain.csv",row.names = F)
predict_c5.0= predict(c5.0_moldel,trainchurnData[,-14])
conf.train_c5<-table(trainchurnData$TARGET,predict_c5.0)
TP <- conf.train_c5[1,1] 
TN <- conf.train_c5[2,2]
FN <- conf.train_c5[1,2]
FP <- conf.train_c5[2,1]
accuracy <- ((TP+TN)/(TP+TN+FN+FP))*100
cat("accuracy is --->", accuracy)
sensitivity= (TP/(TP+FN))*100  
cat("sensitivity is --->", sensitivity) 
specificity = (TN/(TN+FP))*100  
cat("specificity is --->", specificity)
write.csv(rbind(c("Sensitivity=",sensitivity),
                c("Specificity=",specificity),
                c("Accuracy=",accuracy)),"DT_churn_train_metrics.csv",row.names = FALSE)

#######Test Phase############
rval_new <- data.frame(testchurnData,response = predict(modparty, newdata=testchurnData[,-14],type = "response"))
rval_new$prob<-predict(modparty,newdata = testchurnData[,-14],type = "prob")
rval_new$rule <- rls[as.character(predict(modparty,newdata = testchurnData[,-14],type="node"))]
write.csv(rval_new,"churntestallrulestest.csv",row.names = F)
predict_c5.0= predict(c5.0_moldel,testchurnData[,-14])
conf.test_c5<-table(testchurnData$TARGET,predict_c5.0)
TP <- conf.test_c5[1,1] 
TN <- conf.test_c5[2,2]
FN <- conf.test_c5[1,2]
FP <- conf.test_c5[2,1]

accuracy <- ((TP+TN)/(TP+TN+FN+FP))*100
cat("accuracy is --->", accuracy)
sensitivity= (TP/(TP+FN))*100  
cat("sensitivity is --->", sensitivity) 
specificity = (TN/(TN+FP))*100  
cat("specificity is --->", specificity)
write.csv(rbind(c("Sensitivity=",sensitivity),
                c("Specificity=",specificity),
                c("Accuracy=",accuracy)),"DT_churn_test_metrics.csv",row.names = FALSE)

w = table(as.factor(rval_new$rule))
t = as.data.frame(w)
colnames(t)[1]<-"Rule"
newdata1<-t[order(-t$Freq),]
write.csv(newdata1,"churn_test_both_rules_freq.csv",row.names=FALSE)

w=table(rval_new$rule[rval_new$TARGET=="churner"&rval_new$response=="churner"])
t=as.data.frame(w)
colnames(t)[1]<-"Rule"
newdata1<-t[order(-t$Freq),]
write.csv(newdata1,"churn_test_churnonly_rules_freq.csv",row.names=FALSE)

w=table(rval_new$rule[rval_new$TARGET=="nonchurner"&rval_new$response=="nonchurner"])
t=as.data.frame(w)
colnames(t)[1]<-"Rule"
newdata1<-t[order(-t$Freq),]
write.csv(newdata1,"churn_test_nonchurnonly_rules_freq.csv",row.names=FALSE)


#Model-2: Logistic regression
md= glm(trainchurnData$TARGET~., data = trainchurnData, family=binomial(link = "logit"))
predict_glm=predict(md, testchurnData, type="response")
allpredictions=data.frame(testchurnData,c("churner","nonchurner")[predict_glm])
colnames(allpredictions)[length(allpredictions)]<-"pred.TARGET"
write.csv(allpredictions,file = "LR_all_churn_predictions.csv",row.names = FALSE)
cat("Logistic Regression predictions are written to file...")

conf.test_glm<- table(testchurnData$TARGET,allpredictions$pred.TARGET)
TP <- conf.test_glm[1,1] 
TN <- conf.test_glm[2,2]
FN <- conf.test_glm[1,2]
FP <- conf.test_glm[2,1]
accuracy <- ((TP+TN)/(TP+TN+FN+FP))*100
cat("accuracy is --->", accuracy)
sensitivity= (TP/(TP+FN))*100  
cat("sensitivity is --->", sensitivity) 
specificity = (TN/(TN+FP))*100  
cat("specificity is --->", specificity)
write.csv(rbind(c("Sensitivity=",sensitivity),
            c("Specificity=",specificity),
            c("Accuracy=",accuracy)),"LR_metrics.csv",row.names = FALSE)


library(e1071)
svm_moldel= svm(trainchurnData$TARGET~.,data = trainchurnData, kernel="radial", cost=10)
predict_svm= predict(svm_moldel,testchurnData)
allpredictions=data.frame(testchurnData,c("churner","nonchurner")[predict_svm])
colnames(allpredictions)[length(allpredictions)]<-"pred.TARGET"
write.csv(allpredictions,file = "SVM_all_churn_predictions.csv",row.names = FALSE)
cat("SVM predictions are written to file...")

conf.test_svm<- table(testchurnData$TARGET,predict_svm)
TP <- conf.test_svm[1,1] 
TN <- conf.test_svm[2,2]
FN <- conf.test_svm[1,2]
FP <- conf.test_svm[2,1]
accuracy <- ((TP+TN)/(TP+TN+FN+FP))*100
cat("accuracy is --->", accuracy)
sensitivity= (TP/(TP+FN))*100  
cat("sensitivity is --->", sensitivity) 
specificity = (TN/(TN+FP))*100  
cat("specificity is --->", specificity)
write.csv(rbind(c("Sensitivity=",sensitivity),
                c("Specificity=",specificity),
                c("Accuracy=",accuracy)),"SVM_metrics.csv",row.names = FALSE)

#navie bayes 
model <- naiveBayes(trainchurnData$TARGET~.,data = trainchurnData)
predict_naive=predict(model, testchurnData)
allpredictions=data.frame(testchurnData,c("churner","nonchurner")[predict_naive])
colnames(allpredictions)[length(allpredictions)]<-"pred.TARGET"
write.csv(allpredictions,file = "Naive_all_churn_predictions.csv",row.names = FALSE)
cat("Naive Bayes predictions are written to file...")

conf.test_naive<- table(testchurnData$TARGET,allpredictions$pred.TARGET)
TP <- conf.test_naive[1,1] 
TN <- conf.test_naive[2,2]
FN <- conf.test_naive[1,2]
FP <- conf.test_naive[2,1]
accuracy <- ((TP+TN)/(TP+TN+FN+FP))*100
cat("accuracy is --->", accuracy)
sensitivity= (TP/(TP+FN))*100  
cat("sensitivity is --->", sensitivity) 
specificity = (TN/(TN+FP))*100  
cat("specificity is --->", specificity)
write.csv(rbind(c("Sensitivity=",sensitivity),
                c("Specificity=",specificity),
                c("Accuracy=",accuracy)),"naive_metrics.csv",row.names = FALSE)

#random forest
library(randomForest)
Churn_RF<- randomForest(trainchurnData$TARGET~.,data = trainchurnData,ntree=1000,mtry = 4)
importance(Churn_RF)
predict_RF=predict(Churn_RF,testchurnData)
allpredictions=data.frame(testchurnData,c("churner","nonchurner")[predict_RF])
colnames(allpredictions)[length(allpredictions)]<-"pred.TARGET"
write.csv(allpredictions,file = "RF_all_churn_predictions.csv",row.names = FALSE)
conf.test_rf<- table(testchurnData$TARGET,allpredictions$pred.TARGET)
TP <- conf.test_rf[1,1] 
TN <- conf.test_rf[2,2]
FN <- conf.test_rf[1,2]
FP <- conf.test_rf[2,1]
accuracy <- ((TP+TN)/(TP+TN+FN+FP))*100
cat("accuracy is --->", accuracy)
sensitivity= (TP/(TP+FN))*100  
cat("sensitivity is --->", sensitivity) 
specificity = (TN/(TN+FP))*100  
cat("specificity is --->", specificity)
write.csv(rbind(c("Sensitivity=",sensitivity),
                c("Specificity=",specificity),
                c("Accuracy=",accuracy)),"RF_metrics.csv",row.names = FALSE)
