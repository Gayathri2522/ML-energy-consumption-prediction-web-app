from flask import Flask, render_template, url_for, request, json
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def model_trainer(vals):
    #reading dataset
    energy = pd.read_csv('KAG_energydata_complete.csv')
    print(energy)

    #scaling the data
    scaler = MinMaxScaler(feature_range=(0,10))
    energy[['T1','RH_1','T2','RH_2','T3','RH_3','T4','RH_4','T5','RH_5','T6','RH_6','T7','RH_7','T8','RH_8','T9','RH_9','T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint','rv1','rv2']] = scaler.fit_transform(energy[['T1','RH_1','T2','RH_2','T3','RH_3','T4','RH_4','T5','RH_5','T6','RH_6','T7','RH_7','T8','RH_8','T9','RH_9','T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint','rv1','rv2']])
    energy_transformed = energy

    #creating feature vector
    feature = energy_transformed.iloc[:, 3:29]
    print(feature.head())

    #splitting into test and train
    X = feature
    y = energy['Appliances']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    #decision regression tree
    clf11 = DecisionTreeRegressor()
    clf11.fit(X_train,y_train)
    y_pred = clf11.predict(vals)
    return "The predicted value is: " + str(y_pred)

app = Flask(__name__)

@app.route("/", methods=['POST','GET'])
def index():
    if request.method == 'POST':
        data = request.form.getlist("values")
        values = [[i for i in data]]
        return values
        return model_trainer(values)
    else:
        return render_template("index.html")


if __name__ =="__main__":
    app.run(debug=True)