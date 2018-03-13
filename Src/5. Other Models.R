#### Paquetes requeridos ####

install.packages("rpart")
install.packages("e1071")
install.packages("ranger")

library(foreach)
library(data.table)
library(dplyr)

library(rpart)
library(e1071)
library(ranger)


train <- readRDS(paste0(path,"dt_train_featured.rds"))
test <- readRDS(paste0(path,"dt_traintest_featured.rds"))

y_test <- test$Poder_Adquisitivo

# Si usáramos el conjunto de variables sin los pca modificaríamos los missing de la variable 
# Socio_Demo_01 por la moda de tal variable en la tabla train, ya que es un factor categórico

t <- table(train$Socio_Demo_01)
moda <- names(t[which(t == max(t))[1]])
set(train, i=which(is.na(train[["Socio_Demo_01"]])), j="Socio_Demo_01", value=moda)
set(test, i=which(is.na(test[["Socio_Demo_01"]])), j="Socio_Demo_01", value=moda)




#### Linear regression ####

fit.lm <- lm(Poder_Adquisitivo ~., data=train)

predict.lm <- predict(fit.lm,test,type="response")


# mae
mae.lm <- sum(abs(predict.lm-y_test))/length(y_test)

# rmse
rmse.lm <- sqrt(sum((y_test-predict.lm)^2)/length(y_test)) 


summary(fit.lm)

hist(predict.lm,breaks=5000,xlim=c(0,100000),freq = FALSE, main = "Distribución de las predicciones"
     ,xlab = "Valores", ylab = "Densidad")
hist(y_test,breaks=5000,xlim=c(0,100000),add=TRUE,col=rgb(0,0.83,0,0.5),freq = FALSE)
legend("topright", c("Realidad", "Predicción"), col = c(1, 3),
       text.col = c("black","green4"),  pch = c(0, 0), bg = "gray90")

mean(predict.lm)
mean(y_test)

sum(predict.lm)
sum(y_test)

#### Generalized linear regression ####

fit.glm <- glm(Poder_Adquisitivo ~., family=Gamma, data=train)
# Parece que no encuentra unos valores iniciales con los que hacer converger el modelo

predict.glm <- predict(fit.glm,test,type="response")

# mae
mae.glm <- sum(abs(predict.glm-y_test))/length(y_test)

# rmse
rmse.glm <- sqrt(sum((y_test-predict.glm)^2)/length(y_test)) 



#### RPart ####

set.seed(42)
fit.rpart <- rpart(Poder_Adquisitivo~.,data=train)

summary(fit.rpart)

predict.rpart <- predict(fit.rpart,test)


# mae
mae.rpart <- sum(abs(predict.rpart-y_test))/length(y_test)

# rmse
rmse.rpart <- sqrt(sum((y_test-predict.rpart)^2)/length(y_test)) 


#### Random Forest with Ranger ####

set.seed(42)
fit.rgr <- ranger(Poder_Adquisitivo ~.,data=train)

predict.rgr <- predict(fit.rgr,test)

# mae
mae.rgr <- sum(abs(predict.rgr$predictions-y_test))/length(y_test)

# rmse
rmse.rgr <- sqrt(sum((y_test-predict.rgr$predictions)^2)/length(y_test)) 



#### SVM ####

fit.svm <- svm(Poder_Adquisitivo~.,data=train, type= "eps-regression")
# Parece que no encuentra unos valores iniciales con los que hacer converger el modelo

predict.svm <- predict(fit.svm,test,type="response")

# mae
mae.svm <- sum(abs(predict.svm-y_test))/length(y_test)

# rmse
rmse.svm <- sqrt(sum((y_test-predict.svm)^2)/length(y_test)) 





