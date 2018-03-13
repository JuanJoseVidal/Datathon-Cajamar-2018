# En este script se realizará un primer contacto con los datos, con un poco de investigación sobre las variables
# y la correcta transformación de éstas, categóricas a factors y numericas a numeric. Creamos así un conjunto
# train y uno test con variable respuesta para poder validar nuestro modelo.

#### Instalación de paquetes ####

install.packages("data.table")
install.packages("dplyr")
install.packages("foreach")
install.packages("ggplot2")
install.packages("xgboost")
install.packages("rpart")
install.packages("e1071")
install.packages("ranger")


#### Paquetes requeridos ####

library(dplyr)
library(data.table)
library(foreach)

#### Lectura de datos ####

path <- "C:/Users/E105042/RDirectory/Projects/Datathon/" # Path donde estén los scripts y donde se vayan 
# guardar las tablas creadas, working directory, vamos.
data <- data.table(fread(paste0(path,"train.txt"),sep = ","))


table(table(data$ID_Customer)) # No hay customers repetidos
data[,ID_Customer:=NULL] # Eliminamos la variable ID_Customer que no aporta información

hist(data$Poder_Adquisitivo,breaks="Scott",xlim=c(0,50000))
# Los valores se comportan como una Gamma, así que ajustaremos a esta distribución



#### Separación del conjunto train y test ####

# Se van a separar en train y test el train inicial para comprovar la precisión de los modelos. Y para hacer el pca.
set.seed(1984)
data <- data[sample(nrow(data)),]

n <- round(nrow(data)*0.9) # Número de muestras para el train
train <- data[1:n,]
test <- data[(n+1):nrow(data),]

factor.names <- data[,c(grep("Ind_Prod_",names(data),value=TRUE),"Socio_Demo_01","Socio_Demo_02")]


#### Categorical Factors ####

# Separamos los factores categóricos de los numéricos para tratarlos por separado


factors <- train[,factor.names,with=FALSE]
factors.test <- test[,factor.names,with=FALSE]

for(i in 1:ncol(factors)){
  print(table(factors[,i,with=FALSE],useNA = "ifany"))
}
# Vemos los valores distintos de cada factor y vemos si hay NA, que sólo aparecen en la 
# variable Socio_Demo_01. 

factors <- factors[, lapply(.SD, as.factor)]
factors.test <- factors.test[, lapply(.SD, as.factor)]

# Guardaremos los factores para después crear un diccionario con los valores transformados, al 
# pasarlos a numéricos para entrenar Boosted Trees
saveRDS(factors, paste0(path,"factors.rds")) 


#### Numeric Factors ####

numerics <- train[,!(names(train) %in% factor.names),with=FALSE]
numerics.test <- test[,!(names(test) %in% factor.names),with=FALSE]

numerics <- numerics[, lapply(.SD, as.numeric)]
numerics.test <- numerics.test[, lapply(.SD, as.numeric)]

for(i in 1:ncol(numerics)){
  print(summary(numerics[,i,with=FALSE]))
}

# Vemos en los distintos summaries las variables que tienen una variación más grande (observando que los máximos
# se apartan mucho de la media). Esto nos indica que hay pocos posibles outliers como habíamos visto anteriormente.

# Haremos un PCA de las variables numéricas del train para añadirlas a la feature selection y ver
# si aparecen significativas. Usamos el subconjunto "numerics", que son las variables
# numéricas que aparecen en la tabla.


dat.pca <- prcomp(numerics %>% select(-Poder_Adquisitivo)) # Análisis de componentes principales sin la variable respuesta
saveRDS(dat.pca,paste0(paste0(path,"pca.rds"))) # Guardamos para después hacer las predicciones del conjunto test

print(dat.pca)
summary(dat.pca)
plot(dat.pca, type = "l") # Vemos que con 7 componentes la varianza se estabiliza


dt.pca <- data.table(predict(dat.pca,numerics))[,1:6] # Dejamos los 6 primeros componentes, que tienen el 95% de explicación de la variabilidad acumulada
dt.pca.test <- data.table(predict(dat.pca,numerics.test))[,1:6] # Dejamos los 6 primeros componentes, que tienen el 95% de explicación de la variabilidad acumulada




#### Data table final ####

# Creamos la tabla final

dt <- data.table(numerics,dt.pca,factors) 
dt <- dt %>% select(Poder_Adquisitivo,everything()) # Colocamos la respuesta la primera variable
saveRDS(dt, paste0(path,"dt_train.rds")) # Guardamos la tabla para no tener que volverla a generar

dt.test <- data.table(numerics.test,dt.pca.test,factors.test) 
dt.test <- dt.test %>% select(Poder_Adquisitivo,everything()) # Colocamos la respuesta la primera variable
saveRDS(dt.test, paste0(path,"dt_traintest.rds")) # Guardamos la tabla para no tener que volverla a generar



rm(list=setdiff(ls(), c("path","factor.names"))) # Eliminamos tablas intermedias excepto el path
gc() # Garbage collector, para limpiar mejor la memoria cuando se borran tablas grandes

