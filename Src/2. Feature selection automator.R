# Este script realiza para cada una de las n iteraciones un entrenamiento de una muestra aleatoria
# de la tabla inicial con extreme gradient boosting trees y guarda en una lista cuáles han sido las
# variables que han sido más significativas, cortando en un 0.5% de ganancia en la explicación de la
# variable respuesta. Después, recolectamos todas las apariciones y vemos cuantas veces han salido cada
# factor, haciendo así una selección de variables robusta. Se realiza para la base de datos con y sin los PCA.
# Los resultados se encuentran al final del script.


#### Paquetes requeridos ####

library(data.table)
library(xgboost)
library(dplyr)
library(foreach)

#### Carga de datos ####

data <- readRDS(paste0(path,"dt_train.rds"))
featured <- list()
n <- 30

# Para quitar los PCA y hacer la feature selection sin ellos
# pca.names <- data[,grep("PC",names(data),value=TRUE)]
# data <- data[,!(names(data) %in% pca.names),with=FALSE]


foreach(i = 1:n) %do% {
  
  set.seed(i) # Semilla aleatoria
  dt_xgboost <- data[sample(nrow(data),size=60000),] # Seleccionamos una muestra aleatoria para el 
# entrenamiento, en este caso de 60000, pero se puede variar según el número de valores de la tabla total.
# La intención es que todos los valores aparezcan aproximadamente más de una vez (lo cuál no tiene porque ocurrir).
  
  

# Arreglamos los datos para que sean compatibles con el algoritmo  
  y <- as.numeric(dt_xgboost$Poder_Adquisitivo) 
  dt_xgboost[,':='(
    Poder_Adquisitivo = NULL
  )]
  
  dt_xgboost <- dt_xgboost[, lapply(.SD, as.numeric)]
  xgb_mat <- xgb.DMatrix(data = as.matrix(dt_xgboost), label = y)

# Parámetros básicos, regresión gamma debido a que lo hemos visto antes en el histograma
# rmse como medida tradicional para regresión, aunque después calcularemos el mae y compararemos.
  xgb_params <- list(
    "eta" = 0.05
    ,"objective" = "reg:gamma"
    ,"eval_metric" = "rmse"
    ,"booster" = "gbtree"
  )
  
# Entrenamos 500 rondas cada vez
  model_xgb <- xgboost(
    params = xgb_params,
    data=xgb_mat,
    nrounds = 500,
    print_every_n = 50
  )
  
# Sacamos la importancia de las variables usadas
  importance_frame <- xgb.importance(
    model = model_xgb  ,feature_names = names(dt_xgboost)
  )

# Guardamos las que superen el 0.5% de ganancia en la explicación de la respuesta (valor seleccionado por 
# otras experiencias profesionales similares, aunque abierto a debate)
  featured[[i]] <- importance_frame$Feature[importance_frame$Gain>0.005]
  
  gc() # Recolectamos basura para agilizar el procesamiento y gestionar correctamente la memoria
  cat(paste("Finalizada iteración", i,"\n")) # Sacamos info por pantalla para observar el desarrollo
}

saveRDS(featured,paste0(path,"featured.rds")) # Guardamos la lista de apariciones




featured <- readRDS(paste0(path,"featured.rds"))

# Hacemos una cuenta de las apariciones de cada modelo entrenado
featured_unl <- unlist(featured) 
group <- plyr::count(featured_unl)
colnames(group) <- c("variable","apariciones")

# Creamos el vector de las variables que se seleccionarán para entrenar el modelo final.
as.vector(group$variable[group$apariciones>=10])




#### RESULTADOS ####

# Usando los pca y con 30 iteraciones
featured.pca <- c(
  "Imp_Cons_03"   ,"Imp_Cons_08"  , "Imp_Cons_09" ,  "Imp_Sal_07"   , "Imp_Sal_08"  ,  "Imp_Sal_09"  ,  "Imp_Sal_10"  , 
  "Imp_Sal_19"    ,"Imp_Sal_21"   , "Ind_Prod_01" ,  "Num_Oper_06"  , "PC1"         ,  "PC2"         ,  "PC3"         , 
  "Socio_Demo_01" ,"Socio_Demo_03"
)

# Vemos que las tres primeras variables del PCA ha salido muy significativas como explicación de las variables,
# cosa que es lógica, y también cabe remarcar la aparición de variables cuantitativas
# referentes a los salarios de distintos productos financieros

# Sin usar los pca y con 30 iteraciones
featured.sinpca <- c(
"Imp_Cons_03"  , "Imp_Cons_08"  , "Imp_Cons_09"   ,"Imp_Sal_07"  ,  "Imp_Sal_08"   , "Imp_Sal_09"   ,
"Imp_Sal_10"   , "Imp_Sal_19"   , "Imp_Sal_20"    ,"Imp_Sal_21"  ,  "Ind_Prod_01"  , "Num_Oper_05"  ,
"Num_Oper_06"  , "Socio_Demo_01", "Socio_Demo_03"
)
# Por buenas prácticas se ha hecho una feature selection con las variables sin los pca, ya que se 
# tiene en cuenta que la implementación en la empresa de variables sacadas de un pca puede ser más dificultosa
# y nada interpretable. Aún así, se decide usar ambos resultados ya que el rmse que se obtiene es de alrededor
# de 17k y sin el pca de alrededor de 34k. Lo veremos en el siguiente script.

featured <- unique(c(featured.pca,featured.sinpca))


# Filtramos las tablas anteriormente creadas de train y test
train <- readRDS(paste0(path,"dt_train.rds"))
test <- readRDS(paste0(path,"dt_traintest.rds"))

train.feat <- train[,c("Poder_Adquisitivo",featured),with=FALSE]
test.feat <- test[,c("Poder_Adquisitivo",featured),with=FALSE]

saveRDS(train.feat, paste0(path,"dt_train_featured.rds"))
saveRDS(test.feat, paste0(path,"dt_traintest_featured.rds"))

rm(list=setdiff(ls(), c("path","factor.names","featured"))) # Eliminamos tablas intermedias excepto el path
gc() # Garbage collector, para limpiar mejor la memoria cuando se borran tablas grandes



# Guardamos un diccionario de los factores y su futuro valor en el xgb

factors <- readRDS(paste0(path,"factors.rds"))
factors <- factors[,names(factors) %in% featured,with=FALSE]
factors <- data.frame(factors)

dictionary <- list()

for(i in 1:ncol(factors)){
  dictionary[[i]] <- data.frame(valor.real=levels(factors[,i]),valor.xgb=c(1:length(levels(factors[,i]))))
}
names(dictionary) <- names(factors)

saveRDS(dictionary, paste0(path,"dictionary.rds"))


