Predicción de Demanda Energética

Objetivo:
Regresión, predice valor continuo de MWh, con componentes temporales (forecast)

Datos obtenidos de https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption


Estructura

src/data/ make dataset, preprocess
   /features/ build features
   /models/   train, predict, evaluate
   api/  main_api_file
   utils/ helpers

artifacts/ trained models