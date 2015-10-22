# load libraries
require(caret)
require(stringr)
require(rpart)
require(doMC)

# register 4 cores
registerDoMC(cores = 4)

# read in the data
train_raw <- read.csv("./data/raw/train.csv",stringsAsFactors = FALSE)
test_raw <- read.csv("./data/raw/test.csv",stringsAsFactors = FALSE)

########################################################################
## DATA CLEANING
########################################################################
# link training and testing data sets
test_raw$Survived <- NA
all_data <- rbind(train_raw,test_raw)

# fill in missing 'Embarked' data with most common port
all_data$Embarked[all_data$Embarked==''] <- 'S'
# factorize 'Embarked'
all_data$Embarked <- as.factor(all_data$Embarked)
# fill in missing 'Fare' with the median value
all_data$Fare[is.na(all_data$Fare)] <- median(all_data$Fare,na.rm = TRUE)
# create 'Child' feature
all_data$Child[all_data$Age >= 16] <- 0
all_data$Child[all_data$Age < 16] <- 1
# create 'FamilySize' feature
all_data$FamilySize <- all_data$SibSp + all_data$Parch + 1
# create 'Floor' feature
all_data$Floor <- str_match(all_data$Cabin,'^[A-Z]')[,1]
# create 'Title' feature
all_data$Title <- str_match(all_data$Name,"^[A-Za-z\\'\\-\\s]+,\\s([A-Za-z]+)\\.")[,2]
# fill in missing 'Titles'
all_data$Title[is.na(all_data$Title) & all_data$Age >= 16 & all_data$Sex == 'male'] <- 'Mr'
all_data$Title[is.na(all_data$Title) & all_data$Age >= 16 & all_data$Sex == 'female'] <- 'Mrs'
all_data$Title[is.na(all_data$Title) & all_data$Age < 16 & all_data$Sex == 'male'] <- 'Master'
all_data$Title[is.na(all_data$Title) & all_data$Age < 16 & all_data$Sex == 'female'] <- 'Miss'
all_data$Title[grep('Capt|Col|Major',all_data$Title)] <- 'Officer'
all_data$Title[all_data$Title == 'Ms'] <- 'Mrs'
all_data$Title[all_data$Title == 'Mlle'] <- 'Miss'
all_data$Title[all_data$Title == 'Mme'] <- 'Mrs'
all_data$Title[grep('Jonkheer$|Don$',all_data$Title)] <- 'Sir'
all_data$Title[grep('Dona$',all_data$Title)] <- 'Lady'
# factorize 'Title'
all_data$Title <- as.factor(all_data$Title)
# impute missing Ages
predicted_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + Child + FamilySize,
                       data = all_data[!is.na(all_data$Age),], method = "anova")
# set the imputed values for predicted Ages
all_data$Age[is.na(all_data$Age)] <- predict(predicted_age,all_data[is.na(all_data$Age),])
# factorize 'Survived'
all_data$Survived <- factor(all_data$Survived,levels = c('0','1'),labels = c('Died','Survived'))
# split the data
training <- all_data[1:nrow(train_raw),]
testing <- subset(all_data[(nrow(train_raw)+1):nrow(all_data),],select = -Survived)
# set the random seed
set.seed(23432)
# segment the training data
inTrain <- createDataPartition(y = training$Survived,p = 0.80,list = FALSE)
validation <- training[-inTrain,]
training <- training[inTrain,]

################################################################################
## RANDOM FOREST TUNE
################################################################################
# register 4 cores
registerDoMC(cores = 4)
# setup the trainingControl object
tc <- trainControl(method = 'repeatedcv',preProcOptions = c('center','scale'))
# set the random seed
set.seed(23432)
# train the rf model
my_rf <- train(form = as.factor(Survived) ~ as.factor(Pclass) + as.factor(Sex) + SibSp + Parch + Fare + as.factor(Embarked) + FamilySize + as.factor(Title),
               method = 'rf',
               data = training,
               trControl = tc)
# generate validation predictions
my_vpredictions <- predict(my_rf,validation)
# Confusion Matrix
confusionMatrix(my_vpredictions,validation$Survived)
# generate predictions
my_predictions <- predict(my_rf,testing)
# create a data frame of predictions
my_df <- data.frame(PassengerId = testing$PassengerId,Survived = my_predictions)
# output my predictions
write.csv(my_df,file="my_submission.csv",row.names = FALSE)
# register 1 core
registerDoMC(cores = 1)
################################################################################
## C5.0 TUNE
################################################################################
# register 4 cores
registerDoMC(cores = 4)
# setup the trainingControl object
cvCntrl <- trainControl(method = 'repeatedcv', repeats = 3,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE,
                        preProcOptions = c('center','scale'))
# setup tuning grid
grid <- expand.grid(.model = "tree",
                    .trials = c(1:100),
                    .winnow = FALSE)
# set the random seed
set.seed(23432)
# train the rf model
c5Tune <- train(form = as.factor(Survived) ~ as.factor(Pclass) + 
                    as.factor(Sex) + SibSp + Parch + Fare + 
                    as.factor(Embarked) + FamilySize + as.factor(Title),
               method = 'C5.0',
               metric = "ROC",
               data = training,
               tuneGrid = grid,
               trControl = cvCntrl)
# generate validation predictions
c5vpred <- predict(c5Tune,validation)
# confusion matrix
confusionMatrix(c5vpred,validation$Survived)


# generate testing predictions
c5pred <- predict(c5Tune,testing)
# create a data frame of predictions
my_df <- data.frame(PassengerId = testing$PassengerId,Survived = as.double(c5pred)-1)
# output my predictions
write.csv(my_df,file="my_submission.csv",row.names = FALSE)

################################################################################
## SVM TUNE
################################################################################
# setup the trainingControl object
cvCntrl <- trainControl(method = 'repeatedcv', repeats = 3,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE)
# setup tuning grid
grid <- expand.grid(.model = "tree",
                    .trials = c(1:100),
                    .winnow = FALSE)
# set the random seed
set.seed(23432)
# train the rf model
svmTune <- train(form = as.factor(Survived) ~ as.factor(Pclass) + 
                     as.factor(Sex) + SibSp + Parch + Fare + 
                     Embarked + FamilySize + Title,
                 method = 'svmRadial',
                 tuneLength = 9,
                 metric = "ROC",
                 data = training,
                 preProc = c('center','scale'),
                 trControl = cvCntrl)
# generate validation predictions
svmvpred <- predict(svmTune,validation)
# confusion matrix
confusionMatrix(svmvpred,validation$Survived)


# generate testing predictions
svmpred <- predict(svmTune,testing)
# create a data frame of predictions
my_df <- data.frame(PassengerId = testing$PassengerId,Survived = as.double(svmpred)-1)
# output my predictions
write.csv(my_df,file="my_submission.csv",row.names = FALSE)

################################################################################
## SVM FEMALE TUNE
################################################################################
# setup the trainingControl object
cvCntrl <- trainControl(method = 'repeatedcv', repeats = 3,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE)
# set the random seed
set.seed(23432)
# train the rf model
svmTune_females <- train(form = as.factor(Survived) ~ as.factor(Pclass) + 
                             SibSp + Parch + Fare + 
                             Embarked + FamilySize + Title,
                         method = 'svmRadial',
                         tuneLength = 9,
                         metric = "ROC",
                         data = training[training$Sex == 'female',],
                         preProc = c('center','scale'),
                         trControl = cvCntrl)
# generate validation predictions
svmvpred_f <- predict(svmTune_females,validation[validation$Sex=='female',])
# confusion matrix
confusionMatrix(svmvpred_f,validation$Survived[validation$Sex == 'female'])

################################################################################
## SVM MALE TUNE
################################################################################
# setup the trainingControl object
cvCntrl <- trainControl(method = 'repeatedcv', repeats = 3,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE)
# set the random seed
set.seed(23432)
# train the rf model
svmTune_males <- train(form = as.factor(Survived) ~ as.factor(Pclass) + 
                             SibSp + Parch + Fare + 
                             Embarked + FamilySize + Title,
                         method = 'svmRadial',
                         tuneLength = 9,
                         metric = "ROC",
                         data = training[training$Sex == 'male',],
                         preProc = c('center','scale'),
                         trControl = cvCntrl)
# generate validation predictions
svmvpred_m <- predict(svmTune_males,validation[validation$Sex=='male',])
# confusion matrix
confusionMatrix(svmvpred_m,validation$Survived[validation$Sex == 'male'])




# generate testing predictions
svmpred <- predict(svmTune,testing)
# create a data frame of predictions
my_df <- data.frame(PassengerId = testing$PassengerId,Survived = as.double(svmpred)-1)
# output my predictions
write.csv(my_df,file="my_submission.csv",row.names = FALSE)



