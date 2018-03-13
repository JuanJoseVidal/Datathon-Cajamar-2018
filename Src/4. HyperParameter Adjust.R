# En este script utilizaremos cross-validation para ver cuáles son los parámetros que deberíamos de modificar
# en el entrenamiento del booster.


#### Búsqueda de parámetros para mejorar la métrica

searchGridSubCol <- expand.grid(gamma=c(0,2),
                                maxdepth = c(4,6,12),
                                max_delta_step = c(0,3))


ErrorsHyperparameters <- apply(searchGridSubCol, 1, function(parameterList){

  set.seed(42)
  
  gamma <- parameterList[["gamma"]]
  maxdepth <- parameterList[["maxdepth"]]
  max_delta_step <- parameterList[["max_delta_step"]]
  
  xgboostModelCV <- xgb.cv(data =  xgb_mat, nrounds = 100, nfold = 5, showsd = TRUE, 
                           verbose = TRUE, "eval_metric" = "rmse", "gamma" = gamma,
                           "objective" = "reg:linear", "max.depth" = maxdepth, "eta" = 0.1,                           
                           "subsample" = 0.8, "colsample_bytree" = 0.8,"booster" = "gbtree",
                           "max_delta_step" = max_delta_step, "print_every_n" = 5)
  
  #Save rmse of the last iteration
  test_mae <- tail(xgboostModelCV$evaluation_log$test_rmse_mean, 1)
  
  cat("Final de la iteración.","\n")
  
  return(c(test_mae, gamma,maxdepth,max_delta_step))
})



rownames(ErrorsHyperparameters) <- c("test_mae", "gamma","maxdepth","max_delta_step")
ErrorsHyperparameters

# Incrementar el max_delta_step produce que se qeden quietas las predicciones y no avance el entrenamiento
# Cambiar la gamma no es significativo en el mae
# Observamos como modificar la profundidad de los árboles mejora el mean absolute error (mae) 
# del test, por lo que modificaremos est parámetro y volveremos a entrenar 
# (se ha decidido por experiencia usaruna profundidad máxima de los árboles de 4)

