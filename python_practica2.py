###############################################################
#PRÁCTICA 2
###############################################################
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

#FUNCIONES
#################################################################
#Función necesaria para crear retardos temporales en la muestra de datos
def lags(new_data, look_back, pred_col, dim, names):
    indice = new_data.index
    t = new_data.copy()
    t['id'] = range(0, len(t))
    # t = t.iloc[:-look_back, :]
    t = t.iloc[look_back:, :]
    t.set_index('id', inplace=True)
    pred_value = new_data.copy()
    pred_value = pred_value.iloc[:-look_back, pred_col]
    pred_value.columns = names[pred_col]
    pred_value = pd.DataFrame(pred_value)

    pred_value['id'] = range(1, len(pred_value) + 1)
    pred_value.set_index('id', inplace=True)
    final_df = pd.concat([t, pred_value], axis=1)
    indice = np.delete(indice, 0)
    final_df.index = indice

    return final_df

#Función para escalar datos a mano
def minmax_norm(data):
    return (data - data.min()) / ( data.max() - data.min())


########################################################################################################################
data= pd.read_csv(r'C:\Users\migue\Documents\Doctorado\Clases\GET\Practicas_ML\biblioteca_data.csv', sep=';', decimal=',')
data= pd.read_csv(r'C:\Users\migue\Documents\Doctorado\Clases\GET\Practicas_ML\biblioteca_data.csv')



print(data)

#Análisis exploratorio datos
data_copy = data.copy()
print(data_copy)
print(data_copy.shape)
print(data_copy.describe(include='all'))
print(data_copy.describe())
print(data_copy.value_counts('time'))
print(data_copy.corr())



#ELECCIÓN INPUTS MODELO
data_copy.index = data_copy.iloc[:,0]
data_copy = data_copy.drop('time', axis=1)
#data_copy = data_copy.drop(data_copy.columns[0], axis=1)

#Correlaciones (principalmente con la variables dependiente)
correlations = data_copy.corr().iloc[:,0]
ind =  np.where(abs(correlations)>0.25)[0]
print(data_copy.columns[ind])


#Variables temporales
dates = data_copy.index
dates = pd.to_datetime(dates)

hour = pd.DataFrame(dates.hour)
weekday =pd.DataFrame(dates.weekday)
yearday =pd.DataFrame(dates.dayofyear)
temporal = pd.concat([hour, weekday, yearday], axis=1)
temporal.columns = ['Hour', 'Weekday','Yearday']
temporal.index = data_copy.index


data_copy = data_copy.iloc[:,ind[0]]
data_completed = pd.concat([data_copy, temporal], axis=1)

#Creamos retardos temporales en las variables que tenga sentido
data_ret = lags(data_completed,1, np.array([1,2,3,4]), data_completed.shape[0], data_completed.columns)


#Missing values
data_completed.isna().sum()
data_completed.dropna(inplace=True)
data_completed.isna().sum()

#División en X e y
X =data_completed.drop(data_completed.columns[0], axis=1)
y =data_completed.iloc[:,0]

#data_completed.loc['Thermal_demand']
plt.plot(y) #cuidado zonas en torno a cero. Posible errores de medición

#Cogeremos los  3 primeros meses
months = dates.month
st = np.where(months==3)[0][len(np.where(months==3)[0])-1]
y = y.iloc[range(st)]
X = X.iloc[0:st]


#Escalado datos
from sklearn.preprocessing import MinMaxScaler
scalar_X = MinMaxScaler(feature_range=(0,1))
scalar_y = MinMaxScaler(feature_range=(0,1))


scalar_y.fit(pd.DataFrame(y))
y =scalar_y.transform(pd.DataFrame(y))

X_scaled = minmax_norm(X)

#Separación de la muestra en training, test y validación

dates = X.index
dates = pd.to_datetime(dates)
months = dates.month
days = dates.day
y=pd.DataFrame(y)
#Test (10 días del tercer mes)
start1 = np.where((months==3)& (days==11))[0][0]
end1 = np.where((months==3) & (days==20))[0][len(np.where((months==3) & (days==20))[0])-1]
y_test = y.iloc[start1:end1]
X_test = X.iloc[start1:end1]

#Validation (10 últimos días del tercer mes)
start2 = np.where((months==3) & (days==21))[0][0]
end2= np.where(months==3)[0][len(np.where(months==3)[0])-1]
y_val = y.iloc[start2:end2]
X_val = X.iloc[start2:end2]

#Training (resto)
y_train = y.drop(range(start1, len(y)))
X_train = X.drop(X.index[range(start1, len(X))])

#MACHINE LEARNING
#Librerías
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
from sklearn.metrics import r2_score


#Arquitectura de la red
NN_model = Sequential()
# The Input Layer :
NN_model.add(Dense(X_train.shape[1], kernel_initializer='normal', input_dim=X_train.shape[1], activation='relu'))
# The Hidden Layers :
NN_model.add(Dense(100, kernel_initializer='normal', activation='relu'))
# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))
# Compile the network :
NN_model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
NN_model.summary() #resumen del modelo y sus parámetros
# Checkpoitn callback
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
# Train the model
model = NN_model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), batch_size=64, callbacks=[es, mc])
############################################################################
y_pred = NN_model.predict(X_val)
y_pred = scalar_y.inverse_transform(y_pred)
y_val = scalar_y.inverse_transform(y_val)

cv_rmse = np.sqrt(metrics.mean_squared_error(y_val, y_pred)) / np.nanmean(scalar_y.inverse_transform(y))
nmbe = np.mean(y_val - y_pred) / np.nanmean(scalar_y.inverse_transform(y))
print('El CV(RMSE) de predicción en la muestra de validación es', cv_rmse)
print('El NMBE de predicción en la muestra de validación es', nmbe)


#GRAFICOS
plt.plot(y_val, label='Measured', color='black')
plt.plot(y_pred, label='Predicted', color='red')
plt.ylim(0,80)
plt.legend()

#Histograma
residuos = y_val - y_pred
plt.figure()
plt.hist(residuos)
plt.title('Histogram')
#Gráfico líneas

#Diagrama de dispersión


#ANÁLISIS RESIDUOS

#Histograma


#Diagrama de cajas









