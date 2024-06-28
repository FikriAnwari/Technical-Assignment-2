import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#Sampai line 37 ini dipakai buat menyiapkan data
#data train disini memakai data yang TWF nya bernilai 1 seluruhnya dikurangi 5 karena 5 barisnya 
#dipakai buat data test untuk TWF yang bernilai 0 diambil random 20
#data test disini memakai data yang TWF nya bernilai 1 sebanyak 5 baris untuk TWF nya yang bernilai 0
#diambil juga 5 baris  
#memanggil dataset
dataset = pd.read_csv('ai4i2020.csv')

# Membersihkan nama kolom
dataset.columns = dataset.columns.str.replace('[', '').str.replace(']', '')

# Pisahkan data dengan TWF == 1 dan TWF == 0
data_twf_1 = dataset[dataset['TWF'] == 1]
data_twf_0 = dataset[dataset['TWF'] == 0]

# Ambil semua data dengan TWF == 1 untuk pelatihan (kecuali 5 baris terakhir untuk pengujian)
train_data_twf_1 = data_twf_1.iloc[:-5]

# Ambil 50 data acak dengan TWF == 0 untuk pelatihan (kecuali 5 baris terakhir untuk pengujian)
train_data_twf_0 = data_twf_0.iloc[:-5].sample(n=20, random_state=42)

# Gabungkan data untuk pelatihan
train_dataset = pd.concat([train_data_twf_1, train_data_twf_0])
train_dataset_pd = pd.DataFrame(train_dataset)

# Siapkan data untuk pengujian
test_data_twf_1 = data_twf_1.iloc[-5:]
test_data_twf_0 = data_twf_0.iloc[-5:]
test_dataset = pd.concat([test_data_twf_1, test_data_twf_0])
test_dataset_pd = pd.DataFrame(test_dataset)














#prosesing dataset
X_train = np.array(train_dataset[['Air temperature K', 'Process temperature K', 'Rotational speed rpm', 'Torque Nm', 'Tool wear min']])
y_train = np.array(train_dataset['TWF'])

X_train_pd = pd.DataFrame(X_train)
y_train_pd = pd.DataFrame(y_train)
X_train_pd.columns = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear min']
y_train_pd.columns = ['Tool Wear Failure']
print("\n")
print("---------------------------------------TRAIN DATASET---------------------------------------------------------")
print(X_train_pd, "\n")
print(y_train_pd, "\n")
print("-------------------------------------------------------------------------------------------------------------")
print("\n\n")


X_test = np.array(test_dataset[['Air temperature K', 'Process temperature K', 'Rotational speed rpm', 'Torque Nm', 'Tool wear min']])
y_test = np.array(test_dataset['TWF'])
X_test_pd = pd.DataFrame(X_test)
y_test_pd = pd.DataFrame(y_test)
X_test_pd.columns = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear min']
y_test_pd.columns = ['Tool Wear Failure']
print("---------------------------------------TEST DATASET----------------------------------------------------------")
print(X_test_pd, "\n")
print(y_test_pd, "\n")
print("-------------------------------------------------------------------------------------------------------------")
print("\n\n")










#traning model LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# y_pred = abs(y_pred)
print(f'Y hasil prediksi LinearRegression: \n{y_pred}\n')

#traning model RandomForest
model2 = RandomForestClassifier()
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print(f'Y hasil prediksi RandomForestClassifier: \n{y_pred2}\n')
print(f'Y Asli dari dataset: \n{y_test}\n')

print(f'r_squared LinearRegression: \n{r2_score(y_test, y_pred)}\n')
print(f'r_squared RandomForestClassifier: \n{r2_score(y_test, y_pred2)}\n')
print("\n\n\n")








#ini bagian yang mealkukan prediksi lewat input
#User input
while True:
    print("-------------------------------------------------------------------------------------------------------------")
    print("Masukkan input: Air temperature <spasi> Process temperature <spasi> Rotational speed <spasi> Torque <spasi> Tool wear, \n")
    X_input = input("--> ")

    # Konversi input menjadi list float
    X_input_array = [float(x) for x in X_input.split()]

    # Memprediksi nilai menggunakan model
    y_input_predRF = model2.predict([X_input_array])
    y_input_predLG = model.predict([X_input_array])

    print("Ini adalah hasil TWF (Tool Wear Failures) prediksi menggunakan RandomForest:")
    print(y_input_predRF)

    print("Ini adalah hasil TWF (Tool Wear Failures) prediksi menggunakan LinearRegression:")
    print(y_input_predLG, "\n\n")

    #logika untuk keluar dari loop
    print("Lakukan Prediksi lagi (y/n)")
    ver = input("-->")
    if(ver == 'n'):
        print("*EXITING PROGRAM")
        break


