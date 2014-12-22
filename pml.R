# for this classification problem, we intend to predict the type of actions through human sensor data
# I planned to use the following algorithms: decision trees, bagging+decision trees, random forest, and boosting etc. depending on the time
# the steps are: 1) preprocessing the data: scale, transfering factors to numbers if necessary
# exploratory data analysis: plotting, correlation, PCA
# try the decision tree method first and note to use CV method
library(caret)
library(ggplot2)

##remove the NAs variables first
# data is the traning set
data <- read.csv("pml-training.csv", na.string = "NA")
data0 <- read.csv("pml-training.csv", na.string = "")
#data99 is the test set
data99 <- read.csv("pml-testing.csv", na.string = "NA")

remove_NA <- function(data){
    col_num <- dim(data)[2]
    row_num <- dim(data)[1]
    na_col <- c()
    for (i in 1:col_num) {
        if (sum(is.na(data[,i])) > 0.5*row_num) na_col <- c(na_col,i) 
    }
    na_col
}

#pml is the training dataset after removing NA
pml <- data[, -c(remove_NA(data0), remove_NA(data))]
# pml99 is the testing dataset after removing NA
pml99 <- data99[, -c(remove_NA(data0), remove_NA(data))]
#remove data0 
rm(list= "data0")

###change some variable's names
colnames(pml)[1] <- "row_num"
colnames(pml)[dim(pml)[2]] <- "class"

colnames(pml99)[1] <- "row_num"
colnames(pml99)[dim(pml99)[2]] <- "class"

##transform the time to POSIXt class
pml$cvtd_timestamp <-strptime(as.character(pml$cvtd_timestamp), format = "%m/%d/%Y %H:%M")
leftoverDate <- strptime(as.character(data$cvtd_timestamp[which(is.na(pml$cvtd_timestamp))]), format = "%d/%m/%Y %H:%M")
pml$cvtd_timestamp[which(is.na(pml$cvtd_timestamp))] <-leftoverDate
##transfer the time and date variable to numeric for further exploration
pml$cvtd_timestamp <- as.numeric(pml$cvtd_timestamp)

#########99 means testing set
pml99$cvtd_timestamp <-strptime(as.character(pml99$cvtd_timestamp), format = "%m/%d/%Y %H:%M")
leftoverDate99 <- strptime(as.character(data99$cvtd_timestamp[which(is.na(pml99$cvtd_timestamp))]), format = "%d/%m/%Y %H:%M")
pml99$cvtd_timestamp[which(is.na(pml99$cvtd_timestamp))] <-leftoverDate99
##transfer the time and date variable to numeric for further exploration
pml99$cvtd_timestamp <- as.numeric(pml99$cvtd_timestamp)
#dim(pml)
#[1]19622    60



#################################################################
###########################Section: Exploratory data analysis and  Pre_processing the data
#Creating Dummy Variables
#Zero- and Near Zero-Variance Predictors
#Identifying Correlated Predictors
#Linear Dependencies
#Centering and Scaling
#Imputation
#Transforming Predictors
#Class Distance Calculations
#BoxCox, PCA, Imputation, center and scaling, linear dependency, coorrelated predictors, zero-variance predictor, dummy variables


ggplot(pml, aes(x=pml[,1], y=class, colour=class)) +  geom_point()
##so the row_num is not one variable that should be included; is there any other variabls?......
##remove the row_num variable
pml99 <- pml99[,-1]
pml <- pml[, -1]

#for all numeric and int variables: 1)zero-variance predictor 2) imputation 3)boxcox if distribution wierd 4) center and scaling 5) coorrelated predictors 6)linear dependency 7)PCA and predictor transformation
#for the factor variables: 1)dummy variable 3) imputation 4)about the low frequecy but key factor.  5)the above 1),3)-7)? canNOT be done for factors, only do the first three steps about factor Vs

# sector the pml data into numeric and factor datasets and outcomes
pml_n <- pml[, -which(!(lapply(pml, class) == "integer"| lapply(pml, class) == "numeric"))]
pml_f <- pml[, which(!(lapply(pml, class) == "integer"| lapply(pml, class) == "numeric"))[1:2]]
##to keep it as dataframe: datafram[n]
pml_outcome <- pml[which(!(lapply(pml, class) == "integer"| lapply(pml, class) == "numeric"))[3]]


pml_n99 <- pml99[, -which(!(lapply(pml99, class) == "integer"| lapply(pml99, class) == "numeric"))]
pml_f99 <- pml99[, which(!(lapply(pml99, class) == "integer"| lapply(pml99, class) == "numeric"))[1:2]]
##to keep it as dataframe: datafram[n]
pml_outcome99 <- pml99[which(!(lapply(pml99, class) == "integer"| lapply(pml99, class) == "numeric"))[3]]
##########################################
#### 1)dummy variable
pml_f
#use not full-rank method--this is the default--NOt put one level as intercept
dumm_model<-dummyVars(~., data = pml_f, sep = "_")
pml_f_dum <- as.data.frame(predict(dumm_model, pml_f))


pml_f99[,2] <- (c("yes", rep("no",19)))##not sure why --because it should be 19 here
pml_f_dum99 <- as.data.frame(predict(dumm_model, pml_f99))
pml_f_dum99$new_window_no <- rep(1, 20)
pml_f_dum99$new_window_yes <- rep(0, 20)
#########################################return:::: pml_f_dum

##########4)check about the low frequecy but key factor.--whether the factor values are balanced
lapply(pml_f_dum, sum)
##for the users, there is no problem
## for the new window variable, Note that the YES frequency is quite Low; keep this in mind.
########################################

####################################2)imputation
##should set seed here!!!!!!11
#set.seed(777)
# s <- preProcess(pml_n, method=  "knnImpute", k = 10)
# scale <- preProcess(pml_n, method=  c("center","scale"), k = 10)
# sp <- predict(s, pml_n)
# ss <- predict(scale, pml_n)
# identical(sp,  pml_n)
# identical(sp,  ss)
###see above, after k-mean near and ceneter+scale, there is no missing value for the numeric datas
# s <- preProcess(pml_f_dum, method=  "knnImpute", k = 10)
# scale <- preProcess(pml_f_dum, method=  c("center","scale"), k = 10)
# sp <- predict(s, pml_f_dum)
# ss <- predict(scale, pml_f_dum)
# identical(sp,  pml_f_dum)
# identical(sp,  ss)
###see above, after k-mean near and ceneter+scale, there is no missing value for the factor data
############################################

####1)zero-variance predictor for the numeric Variables only!!!
nzv <- nearZeroVar(pml_n, saveMetrics = TRUE)
sum(nzv$nzv)
###there is no variable have 0 or near zero variance; so this is Good
################################


##### 3)boxcox if distribution wierd 
#plot each, you have to; then check which is not good; then you can apply boxcox or other method....w
#par(mfrow=c(8,8))???
##seq_len(dim(pml_n)[2])
# for (i in 21:30 ){
#     print(i)
#     windows(i)
#     hist((pml_n[[i]]))    
#     
# }
###basic histgram showing that most of the variabls having non-bell like distribution, so we have to transform them!!
set.seed(777)
norm_tranYJ <-preProcess(pml_n, method = "YeoJohnson")
tranYJ_pml_n <- predict(norm_tranYJ, pml_n)
tranYJ_pml_n99 <- predict(norm_tranYJ, pml_n99)
##check  whether the transform data is better now
windows()
##specify the settings below the windows
par(mfrow=c(8,8))
###Error in plot.new() : figure margins too large :::fix it: shorten the margin...
par(mar = rep(1,4))
for (i in seq_len(dim(pml_n)[2])){
    print(i)
    #windows(i)
    hist((tranYJ_pml_n[[i]]))    
    
}
### the following is using the exponential transfer::
# set.seed(777)
# norm_tranEX <-preProcess(pml_n, method = "expoTrans")
# tranEX_pml_n <- predict(norm_tranEX, pml_n)
# ##check  whether the transform data is better now
# windows()
# ##specify the settings below the windows
# par(mfrow=c(8,8))
# ###Error in plot.new() : figure margins too large :::fix it: shorten the margin...
# par(mar = rep(1,4))
# for (i in seq_len(dim(pml_n)[2])){
#     print(i)
#     #windows(i)
#     hist((tranEX_pml_n[[i]]))      
# }
# return  :: tranYJ_pml_n
##so the YJ transformation is better, choose it
######################################################
##### 4) center and scaling 
center_scale <- preProcess(tranYJ_pml_n, method = c("center","scale"))
tranYJ_CS_pml_n <- predict(center_scale, tranYJ_pml_n)
tranYJ_CS_pml_n99 <- predict(center_scale, tranYJ_pml_n99)
#################################return:: tranYJ_CS_pml_n

#####5) coorrelated predictors 
descrCor <- cor(tranYJ_CS_pml_n)
summary(descrCor[upper.tri(descrCor)])
highlyCorDescr <- findCorrelation(descrCor, cutoff = 0.85)
## the findCorrelation return column index that should be removed due to pair correlation
tranYJ_CS_ReCo_pml_n <- tranYJ_CS_pml_n[, -highlyCorDescr]
tranYJ_CS_ReCo_pml_n99 <- tranYJ_CS_pml_n99[, -highlyCorDescr]
descrCor2 <- cor(tranYJ_CS_ReCo_pml_n)
summary(descrCor2[upper.tri(descrCor2)])
#################################return:: tranYJ_CS_ReCo_pml_n
#####6)linear dependency 
findLinearCombos(pml_n)
findLinearCombos(tranYJ_CS_ReCo_pml_n)
####findLinearCombos(pml_f_dum[7:8])
##there is no linear dependent variables
########################return:::: tranYJ_CS_ReCo_pml_n

#####7)PCA and predictor transformation
pca_model <- preProcess(tranYJ_CS_ReCo_pml_n, method = "pca", thresh = 0.95)## is 0.95 as threshold good or bad?
summary(pca_model)
pca_model$numComp
dim(tranYJ_CS_ReCo_pml_n)
tranYJ_CS_ReCo_PCA_pml_n<- predict(pca_model, tranYJ_CS_ReCo_pml_n)
tranYJ_CS_ReCo_PCA_pml_n99<- predict(pca_model, tranYJ_CS_ReCo_pml_n99)


###########################################collect the dataset
pml
# dim(pml)
# [1] 19622    60
#####################have to remove the row numbers!!!!!!!
pml_dum <-  cbind(pml_n, pml_f_dum, pml_outcome)
pml_dum99 <-  cbind(pml_n99, pml_f_dum99, pml_outcome99)
############
pml_pca <-cbind(tranYJ_CS_ReCo_PCA_pml_n,pml_f_dum, pml_outcome)
pml_pca99 <-cbind(tranYJ_CS_ReCo_PCA_pml_n99,pml_f_dum99, pml_outcome99)
# dim(pml_pca)
# [1] 19622    39
pml_remove_corr <-cbind(tranYJ_CS_ReCo_pml_n,pml_f_dum, pml_outcome)
pml_remove_corr99 <-cbind(tranYJ_CS_ReCo_pml_n99,pml_f_dum99, pml_outcome99)
# > dim(pml_remove_corr)
# [1] 19622    60
pml_pca_dum <- cbind(tranYJ_CS_ReCo_PCA_pml_n, pml_f, pml_outcome)
pml_pca_dum99 <- cbind(tranYJ_CS_ReCo_PCA_pml_n99, pml_f99, pml_outcome99)
# > dim(pml_pca_dum)
# [1] 19622    33
#save(pml,pml99,pml_dum,pml_dum99, pml_pca,pml_pca99, pml_remove_corr, pml_remove_corr99, pml_pca_dum,pml_pca_dum99,  file= "processed_data_pml.RData")


###################model fitting####################
####first model is a simple tree model  with n folder CV
##try a simple tree model with cross validation
set.seed(12312)
inTrain <- createDataPartition( y= pml_pca$class, p = 0.9, list = FALSE)
training <- pml_pca[inTrain, ]
testing  <- pml_pca[-inTrain, ]
#identical(sort(unique(c(testing[[1]], training[[1]]))), unique(pml[[1]]))-----so simple bootstrap sampling called in caret here is sample without replacement, which is a little bit confusing...
fitControl <- trainControl(## repeated 10-fold CV
    method = "repeatedcv",
    number = 10,
    ## repeated  3 times
    repeats = 3)

set.seed(12023)
pca_mod_rpart_fit <- train(class~., data = training, method = "rpart" , trControl = fitControl)

testing_pca_rpart_predict <- predict(pca_mod_rpart_fit, testing)

sum(testing_pca_rpart_predict == testing$class)/length(testing_pca_rpart_predict)

pml_pca99_pca_rpart_predict <- predict(pca_mod_rpart_fit, pml_pca99)
#results:
#C A A C C D C C A A A C A A C E A C C D
########################################################3

##two models: random forest and boosting with tree(too much computation there)
#remember to set seed!!!!!
set.seed(12312)
inTrain <- createDataPartition( y= pml_pca$class, p = 0.9, list = FALSE)
training <- pml_pca[inTrain, ]
testing  <- pml_pca[-inTrain, ]

set.seed(1770)
pca_mod_rf_fit <- train( class ~ .,data = training, method = "rf")
#save(pca_mod_rf_fit, file= "pca_mod_rf_fit_onlyTrainingSet.RData")
#save()
#pca_mod_rf_fit<-local(get(load("./pca_mod_rf_fit_onlyTrainingSet.RData")))


testing_pca_rf_predict <- predict(pca_mod_rf_fit, testing)
sum(testing[,"class"] == testing_pca_rf_predict)/length(testing_pca_rf_predict)
########there is overfitting in there
pml_pca99_pca_rf_predict <- predict(pca_mod_rf_fit, pml_pca99)
# B(wrong) A A(wrong) A(wrong) A E D B A A B C B A E E A B B B
################################################################


#########################the following models are not reported in the project report.############
##################################################################################################
##random forest
set.seed(12312)
inTrain <- createDataPartition( y= pml_remove_corr$class, p = 0.9, list = FALSE)
training <- pml_remove_corr[inTrain, ]
testing  <- pml_remove_corr[-inTrain, ]
fitControl <- trainControl(## repeated 10-fold CV
    method = "repeatedcv",
    number = 10,
    ## repeated  3 times
    repeats = 1)


set.seed(17701)
remove_corr_mod_rf_fit <- train( class ~.,data = training, method = "rf", trControl = fitControl)
#save(pca_mod_rf_fit, file= "pca_mod_rf_fit.RData")
#save()
#pca_mod_rf_fit<-local(get(load("./pca_mod_rf_fit.RData")))


pred_results_remove_corr_rf_testing <- predict(remove_corr_mod_rf_fit, testing)
pred_results_remove_corr_rf_99 <- predict(remove_corr_mod_rf_fit,  pml_remove_corr99)
# pred_results_rf99
# [1] B A A A A A A B A A A A B A B A A B B B
#
#sum(pml_remove_corr[,"class"] == pred_results_remove_corr_rf)/length(pred_results_remove_corr_rf)
########there is overfitting in the model





####the boosting with tree model should be good, but too much computationnal cost::
pca_mod_gbm_fit <- train( class ~.,data = pml_pca, method = "gbm", verbose = T)




#######useing the original non-processing dataset####################
######try a simple tree model with cross validation###################
set.seed(12312)
inTrain <- createDataPartition( y= pml$class, p = 0.9, list = FALSE)
training <- pml[inTrain, ]
testing  <- pml[-inTrain, ]
#identical(sort(unique(c(testing[[1]], training[[1]]))), unique(pml[[1]]))-----so simple bootstrap sampling called in caret here is sample without replacement, which is a little bit confusing...
#training <- pml[inTrain, ]
#testing  <- pml[-inTrain, ]

fitControl <- trainControl(## repeated 10-fold CV
    method = "repeatedcv",
    number = 10,
    ## repeated  5 times
    repeats = 5)

set.seed(12023)
tree_mod_rpart_fit <- train(class~., data = training, method = "rpart" , trControl = fitControl)
tree_predict_result_testing <- predict(tree_mod_rpart_fit, testing)
sum(tree_predict_result_testing == testing$class)/length(tree_predict_result_testing)
###########################################
###################for the remove correlation datasets

set.seed(12312)
inTrain <- createDataPartition( y= pml_remove_corr$class, p = 0.9, list = FALSE)
training <- pml_remove_corr[inTrain, ]
testing  <- pml_remove_corr[-inTrain, ]
#identical(sort(unique(c(testing[[1]], training[[1]]))), unique(pml[[1]]))-----so simple bootstrap sampling called in caret here is sample without replacement, which is a little bit confusing...
fitControl <- trainControl(## repeated 10-fold CV
    method = "repeatedcv",
    number = 10,
    ## repeated  3 times
    repeats = 1)
set.seed(12023)
tree_mod_rpart_fit <- train(class~., data = training, method = "rpart" , trControl = fitControl)
tree_predict_result_testing <- predict(tree_mod_rpart_fit, testing)
#tree_predict_result_testing <- predict(tree_mod_rpart_fit, testing)
sum(tree_predict_result_testing == testing$class)/length(tree_predict_result_testing)


###################pml_dum, only processing the factor variables
set.seed(12312)
inTrain <- createDataPartition(y= pml_dum$class, p = 0.9, list = FALSE)
training <- pml_dum[inTrain, ]
testing  <- pml_dum[-inTrain, ]
#identical(sort(unique(c(testing[[1]], training[[1]]))), unique(pml[[1]]))-----so simple bootstrap sampling called in caret here is sample without replacement, which is a little bit confusing...
fitControl <- trainControl(## repeated 10-fold CV
    method = "repeatedcv",
    number = 10,
    ## repeated  3 times
    repeats = 1)

set.seed(12023)
tree_mod_rpart_fit <- train(class~., data = training, method = "rpart" , trControl = fitControl)

tree_predict_result_testing <- predict(tree_mod_rpart_fit, testing)
tree_predict_result_testing99 <- predict(tree_mod_rpart_fit, pml_dum99)
sum(tree_predict_result_testing == testing$class)/length(tree_predict_result_testing)
tree_predict_result_testing99





