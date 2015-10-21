# load libraries
require(caret)

# read in the data
training <- read.csv("./data/raw/train.csv",stringsAsFactors = FALSE)
testing <- read.csv("./data/raw/test.csv",stringsAsFactors = FALSE)

########################################################################
## DATA CLEANING
########################################################################
# link training and testing data sets
testing$Survived <- NA
all_data <- rbind(training,testing)

# fill in missing 'Embarked' data with most common port
all_data$Embarked[all_data$Embarked==''] <- 'S'
# create 'Child' feature
all_data$Child[all_data$Age >= 18] <- 0
all_data$Child[all_data$Age < 18] <- 1
# create 'FamilySize' feature
all_data$FamilySize <- all_data$SibSp + all_data$Parch + 1
# create all_data control
tc <- trainControl(method = "repeatedcv",number = 10)
# seed the random number generator
set.seed(23432)
# impute missing Ages
ageFit <- train(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + FamilySize + Child,
                data = all_data,
                method = "rpart",
                trControl = tc,
                metric = "RMSE")
# set the imputed values for predicted Ages
head(predict(ageFit,all_data))
head(all_data$Age)