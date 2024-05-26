from flask import Flask, render_template,request
import joblib
import io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

model = joblib.load('modelfix.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global upload_df

    if 'csv_file' not in request.files:
        return "No file part"
    
    file = request.files['csv_file']
    
    if file.filename == '':
        return "No selected file"
    
    if file:
        # Baca file CSV
        df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")))
        
        # Buang baris yang mengandung nilai NaN
        df.dropna(inplace=True)
        
        # Simpan DataFrame ke dalam variabel global
        upload_df = df
        
        # Ambil lima baris pertama
        baris5 = df.head()
        return render_template('index.html', data=baris5.to_html())


@app.route('/one_hot', methods=['POST'])
def one_hot():
    global upload_df
    global df_final

    if upload_df is None:
        return "No file"
    
    # Lakukan one-hot encoding
    df_encoded = pd.get_dummies(upload_df)

    df_encoded.rename(columns={'LKS_Alam Sutera': 'Alam Sutera', 'LKS_BSD': 'BSD', 'LKS_Gading Serpong': 'Gading Serpong'}, inplace=True)

    df_encoded[['Alam Sutera', 'BSD', 'Gading Serpong']] = df_encoded[['Alam Sutera', 'BSD', 'Gading Serpong']].astype(int)

    df_final = pd.concat([df_encoded.drop('Harga', axis=1), df_encoded['Harga']], axis=1)

    limadata = df_final.head()

    return render_template ('index.html', preprocessing=limadata.to_html())

@app.route('/modelroute', methods=['POST'])
def modeljadi():

    note = "Model berhasil dibuat."

    return render_template('index.html',note=note)

@app.route('/hasildata', methods=['POST'])
def hasildata():
    global df_final

    if df_final is None:
        return "No data"

    # Memisahkan variabel X dan Y
    X = df_final.drop('Harga', axis=1)
    Y = df_final['Harga']

    # Membagi data menjadi data latih dan data uji
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    # Membuat model Random Forest Regression
    modelrf = RandomForestRegressor(n_estimators=100, random_state=2)
    modelrf.fit(X_train, Y_train)

    # Memprediksi harga menggunakan model
    Y_pred = modelrf.predict(X_test)

    # Mengonversi array ke satu dimensi
    Y_test_flat = Y_test.values.ravel()
    Y_pred_flat = Y_pred.ravel()

    # Konversi data sebenarnya dan prediksi ke dalam format DataFrame
    dataframe = pd.DataFrame({'Data Sebenarnya': Y_test_flat, 'Data Prediksi': Y_pred_flat})

    # Fungsi untuk mengubah nilai menjadi format mata uang Rupiah
    def format_as_rupiah(value):
        return "Rp {:,.2f}".format(value)

    # Mengaplikasikan fungsi format_as_rupiah ke kolom DataFrame
    dataframe['Data Sebenarnya'] = dataframe['Data Sebenarnya'].apply(format_as_rupiah)
    dataframe['Data Prediksi'] = dataframe['Data Prediksi'].apply(format_as_rupiah)
    
    # Mengembalikan HTML tabel hasil
    hasil = dataframe.head(10).to_html(index=False)

    return render_template('index.html', hasil=hasil)

@app.route('/akurasi', methods=['POST'])
def akurasi():
    global df_final

    if df_final is None:
        return "No data"

    # Memisahkan variabel X dan Y
    X = df_final.drop('Harga', axis=1)
    Y = df_final['Harga']

    # Membagi data menjadi data latih dan data uji
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    # Membuat model Random Forest Regression
    modelrf = RandomForestRegressor(n_estimators=100, random_state=2)
    modelrf.fit(X_train, Y_train)

    # Memprediksi harga menggunakan model
    Y_pred = modelrf.predict(X_test)

    # Menghitung akurasi model
    mae = mean_absolute_error(Y_test,Y_pred)

    mape = mean_absolute_percentage_error(Y_test, Y_pred) * 100

    rsquared = metrics.r2_score(Y_test,Y_pred)

    mae_format = "{:.2f}".format(mae)
    mape_format = "{:.2f}".format(mape)
    rsquared_format = "{:.2f}".format(rsquared)

    return render_template('index.html', mae=mae_format, mape=mape_format, rsquared=rsquared_format)

@app.route('/visual', methods=['POST'])
def visual():

    global df_final

    if df_final is None:
        return "No data"

    # Memisahkan variabel X dan Y
    X = df_final.drop('Harga', axis=1)
    Y = df_final['Harga']

    # Membagi data menjadi data latih dan data uji
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    # Membuat model Random Forest Regression
    modelrf = RandomForestRegressor(n_estimators=100, random_state=2)
    modelrf.fit(X_train, Y_train)
    Y_pred = modelrf.predict(X_test)
    Y_test = list(Y_test)

    plt.clf() #bersihkan data visual sebelumnya

    plt.plot(Y_test, color='green', label='Harga asli')
    plt.plot(Y_pred, color='blue', label='harga prediksi')
    plt.title('Harga asli vs Harga Prediksi')
    plt.xlabel('Value')
    plt.ylabel('harga')
    plt.legend()

    plot_path = 'static/plot.png'

    return render_template('index.html',plot_path=plot_path)

@app.route('/prediksi.html')
def prediksi():
    return render_template('prediksi.html')

@app.route('/predict', methods=['POST'])
def predict():

    global df_final
    
    data = {
        'LB': [request.form['LB']],
        'LT': [request.form['LT']],
        'KM': [request.form['KM']],
        'KT': [request.form['KT']],
        'GRS': [request.form['GRS']],
        'Alam Sutera': [request.form.get('Alam_Sutera', 0)],
        'BSD': [request.form.get('BSD', 0)],
        'Gading Serpong': [request.form.get('Gading_Serpong', 0)]
    }

    # prediksi menggunakan model
    predicted_price = model.predict(pd.DataFrame(data))
    
    # Tampilkan hasil prediksi
    return render_template('prediksi.html', predicted_price=predicted_price[0])

if __name__ == '__main__':
    app.run(debug=True)
