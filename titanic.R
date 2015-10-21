# load libraries
require(caret)

# read in the data
training <- read.csv("./data/raw/train.csv",stringsAsFactors = FALSE)
testing <- read.csv("./data/raw/test.csv",stringsAsFactors = FALSE)

# fill in missing 'Embarked' data with most common port
training$Embarked[training$Embarked==''] <- 'S'

# create training control
tc <- trainControl(method = "repeatedcv",number = 10)
# seed the random number generator
set.seed(23432)
# impute missing Ages
ageFit <- train(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked,
                data = training,
                method = "rpart",
                trControl = tc,
                metric = "RMSE")
# set the imputed values for predicted Ages
head(predict(ageFit,training))
head(training$Age)