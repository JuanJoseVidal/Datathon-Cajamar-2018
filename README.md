# Datathon-Cajamar-2018

Este repositorio contiene el código con el que se ha realizado el Datathon de Cajamar, reto Salesforce Predictive Modelling, así como las tablas intermedias creadas y los gráficos obtenidos. Explicamos a continuación como hemos organizado los scripts:


El código para este Datathon se divide en cinco partes, cada una en su respectivo archivo .R, en cada uno se encuentran comentarios sobre lo que se va descubriendo sobre los datos y sobre lo que se va haciendo, a parte de por qué se hace y las conclusiones obtenidas. 

Para una cómoda reproducción de los resultados, se ha añadido una variable llamada path que indica la carpeta del sistema donde se encuentra el código y se donde se generarán los datasets que se irán surgiendo según avance el código.

El formato de los códigos está codificado en UTF-8, por lo que si los acentos de los comentarios no se leen correctamente, sólo habrá que ir a File>Reopen With Encoding... y seleccionar UTF-8.

Pasemos a explicar los cinco scripts:

El primero, llamado Data Wrangling, se dedica a explorar brevemente los datos, busca valores vacíos, que sólo se encuentran en la variable Socio_Demo_01. Así pues, se separan las variables numéricas y categóricas para un tratamiento correcto de los datos. También se realiza un PCA de las variables numéricas para añadir más variables explicativas a los datos.

El segundo, llamado Feature Selection automator, realiza múltiples iteraciones sobre los datos, seleccionando subconjuntos y entrenando xgboosters en ellos guardando en cada iteración las variables que superan un 0.5% de ganancia en la explicación de la variable respuesta, creando así una selección de variables robusta. Generamos así las tabla únicamente con las variables con las que entrenaremos los distintos modelos.

El tercero, llamado XGB modelling modeliza los datos con el algoritmo XGBoost con los parámetros obtenidos en el cuarto script, HyperParameter Adjust. Finalmente se exponen las medidas obtenidas para seleccionar el modelo adecuado, siendo los otros modelos entrenados en el quinto script, Other Models. Las medidas utilizadas son el Mean Squared Error y el Root Mean Squared Error. 

Las conclusiones obtenidas son que el paquete ranger para hacer RandomForests y el XGBoost producen unos resultados similares, pero al tener que hacer imputación de datos missing para la variable Socio_Demo_01 para entrenar los RandomForest, se selecciona el XGBoost como modelo final para realizar la predicción y la entrega ya que no requiere de imputación de variables faltantes y esto en el mundo real puede suponer una gran pérdida de información.
