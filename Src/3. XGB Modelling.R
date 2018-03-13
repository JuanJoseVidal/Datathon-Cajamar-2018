
#### Paquetes requeridos ####

library(data.table)
library(dplyr)
library(xgboost)
library(ggplot2)


#### Lectura de datos ####

dt_xgboost <- data.table(readRDS(paste0(path,"dt_train_featured.rds")))
test_xgboost <- data.table(readRDS(paste0(path,"dt_traintest_featured.rds")))

dictionary <- readRDS(paste0(path,"dictionary.rds"))


#### XGBoost - Preparación de los datos ####

set.seed(1234)

# Arreglamos el conjunto train
y <- dt_xgboost$Poder_Adquisitivo
dt_xgboost[,':='(
  Poder_Adquisitivo = NULL
)]


for (vble in names(dictionary)){
  for(i in nrow(dictionary[[vble]]):1){
    old.value <- as.numeric(as.character(dictionary[[vble]]$valor.real[i]))
    new.value <- dictionary[[vble]]$valor.xgb[i]
    
    levels(dt_xgboost[[vble]])[levels(dt_xgboost[[vble]])==old.value] <- new.value
  }
  dt_xgboost[[vble]] <- as.numeric(as.character(dt_xgboost[[vble]]))
}
xgb_mat <- xgb.DMatrix(data = as.matrix(dt_xgboost), label = y)

# Arreglamos el conjunto test
y_test <- test_xgboost$Poder_Adquisitivo
test_xgboost[,':='(
  Poder_Adquisitivo = NULL
)]


for (vble in names(dictionary)){
  for(i in nrow(dictionary[[vble]]):1){
    old.value <- as.numeric(as.character(dictionary[[vble]]$valor.real[i]))
    new.value <- dictionary[[vble]]$valor.xgb[i]
    
    levels(test_xgboost[[vble]])[levels(test_xgboost[[vble]])==old.value] <- new.value
  }
  test_xgboost[[vble]] <- as.numeric(as.character(test_xgboost[[vble]]))
}


dtest <- xgb.DMatrix(as.matrix(test_xgboost),label=y_test)

watchlist <- list(train=xgb_mat,test=dtest) # Para visualizar los avances del entrenamiento y controlar overfitting


#### XGBoost - Parámetros ####

# Hemos probado con el rmse y avanza más rápido que con el mae, obteniendo resultados similares.
# Para controlar el sobrejajuste usamos el early_stopping rounds y en el momento en el que el error del test
# empiece a subir se detendrá el entrenamiento y se quedará el mejor modelo observado calificado con el test.

xgb_params <- list(
  "eta" = 0.05 # Velocidad de entrenamiento (learning rate)
  ,"objective" = "reg:gamma"
  ,"eval_metric" = "mae"
  ,"booster" = "gbtree"
  ,"max_depth" = 4 # Para saber por que se utilizan estos parámetros, ir al script HyperParameter Adjust
  )

set.seed(42) # Debido a la aleatoriedad del modelo, marcamos una semilla para reproducir los resultados


#### XGBoost - Entrenamiento ####

model_xgb <- xgb.train(
  params = xgb_params,
  data = xgb_mat,
  nrounds = 10000,
  watchlist = watchlist,
  print_every_n = 100,
  early_stopping_rounds = 200
)



# Gracias al early_stopping_rounds el modelo para de entrenar cuando lleva 200 rondas sin mejorar 
# la predicción del test y evitar overfitting. En unas 200 rondas el modelo para de entrenarse.

predict.xgb <- predict(model_xgb,dtest)

# mae
mae.xgb <- sum(abs(predict.xgb-y_test))/length(y_test)

# rmse
rmse.xgb <- sqrt(sum((y_test-predict.xgb)^2)/length(y_test)) 



# Vemos que se ha ajustado a una gamma bastante bien proporcionada
hist(predict.xgb,breaks=5000,xlim=c(0,100000),freq = FALSE, main = "Distribución de las predicciones"
     ,xlab = "Valores", ylab = "Densidad")
hist(y_test,breaks=5000,xlim=c(0,100000),add=TRUE,col=rgb(0,0.83,0,0.5),freq = FALSE)
legend("topright", c("Realidad", "Predicción"), col = c(1, 3),
       text.col = c("black","green4"),  pch = c(0, 0), bg = "gray90")



# Vemos que en valores medios y en sumas a términos de cartera ajusta bastante bien
mean(predict.xgb)
mean(y_test)

sum(predict.xgb)
sum(y_test)

# Visualización de los árboles
xgb.plot.multi.trees(feature_names = names(dt_xgboost),model = model_xgb)


#### CONCLUSIÓN ####

# Hemos probado distintos modelos y se han utilizado dos métricas para comparar los resultados,
# el mean absolute error (mae) y el root mean squared error (rmse), siendo los resultados fácilmente replicables,
# en el script Other Models se encontrarán los entrenamientos de los otros.

#   +------+--------------+-------------+---------------+------------------------+-------------+----------+
#   |      | Linear Model | GLM (Gamma) | Decision Tree | Random Forest (Ranger) |     SVM     |  XGBoost |
#   +------+--------------+-------------+---------------+------------------------+-------------+----------+
#   |  MAE |    5911.06   | No converge |    6420.97    |         4584.76        | No converge |  4334.75 |
#   +------+--------------+-------------+---------------+------------------------+-------------+----------+
#   | RMSE |     31376    | No converge |    31951.09   |        29763.76        | No converge | 30043.20 |
#   +------+--------------+-------------+---------------+------------------------+-------------+----------+
   
# Hemos visto que el único modelo que obtiene unos resultados similares al xgboost es el Random Forest con
# el paquete ranger. En este caso, el xgboost reduce el mae de 4584.76 (rgr) a 4334.75 (xgb), una reducción del 5.5%, mientras que 
# el rmse aumenta de 29763.76 (rgr) a 30043.20 (xgb), aumentando en un 1%. Viéndolo así no tenemos evidencia suficiente
# para decidir qué modelo es mejor que otro. Si se realizan la suma de las predicciones y la media, el ranger 
# se aproxima mucho mejor a los valores generales del test. Aún así, no lo hemos considerado suficiente como para
# razonar la selección de uno u otro.

# Un punto a favor del xgboost es que puede tratar los NA sin problema, mientras que para el 
# Random Forest se necesita hacer imputación de valores vacíos, y puede provocar una pérdida de información
# o información errónea si no se realiza correctamente.

# Finalmente, debido a la fama que el xgboost ha ganado en los últimos años, y que generalmente está aceptado 
# como uno de los mejores algoritmos de Machine Learning, seleccionaremos éste como el modelo a aplicar para 
# la predicción del conjunto test.



#### XGBoost - Guardado y carga ####

xgb.save(model_xgb, paste0(path,"xgb_model_final_mae005.model"))
model_xgb <- xgb.load(paste0(path,"xgb_model_final_mae005.model"))



#### Predicción del conjunto test para submit ####


TEST <- data.table(fread(paste0(path,"test.txt"),sep = ",")) # Cargamos los datos del Test
dat.pca <- readRDS(paste0(path,"pca.rds")) # Cargamos el pca para predecir sus valores

ID <- TEST[["ID_Customer"]] # Guardamos las ID para ponerlas en el submit

TEST.pca <- data.table(predict(dat.pca,TEST))[,1:6] # Predecimos los valores del PCA
TEST_full <- data.table(TEST,TEST.pca) # Los añadimos al test
TEST_full <- TEST_full[,featured,with=FALSE] # Nos quedamos sólo con las variables significativas estudiadas anteriormente

# Transformamos las variables factoriales en números que coincidan con los estudiados en el xgboost
dictionary <- readRDS(paste0(path,"dictionary.rds"))
for (vble in names(dictionary)){
  for(i in nrow(dictionary[[vble]]):1){
    old.value <- as.numeric(as.character(dictionary[[vble]]$valor.real[i]))
    new.value <- dictionary[[vble]]$valor.xgb[i]
    
    levels(TEST_full[[vble]])[levels(TEST_full[[vble]])==old.value] <- new.value
  }
  TEST_full[[vble]] <- as.numeric(as.character(TEST_full[[vble]]))
}


TEST.dtest <- xgb.DMatrix(as.matrix(TEST_full)) # Generamos la xgb.DMatrix ppara poder evaluar el modelo

predict.final <- predict(model_xgb,TEST.dtest) # Predecimos los poderes adquisitivos
ggplot(data.table(predict.final),aes(predict.final)) + 
  geom_histogram(breaks=seq(0, 50000, by = 200), 
                 col="gray49",
                 fill = "orange3") +
  ggtitle("Predicción del conjunto test") +
  labs(x = "Poder Adquisitivo", y = "Frecuencia") +
  theme(panel.background = element_rect(fill = 'white', colour = 'white'),
        panel.grid.major = element_line(colour = "orange"))
# Vemos que ha adaptado la forma de la Gamma


submit <- data.table(ID,predict.final) # Juntamos los ID con las predicciones 
names(submit) <- c("ID_Customer","PA_Est") # Le añadimos los nombres correspondientes a las columnas

fwrite(submit,paste0(path,"Test_Mission.txt")) #Generamos el .txt para enviar




