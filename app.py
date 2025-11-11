import streamlit as st # Por convenção, vamos apelidar o streamlit de st
import pandas as pd # Importando a biblioteca Pandas

# Importando as demais bibliotecas necessárias
import numpy as np
import altair as alt

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

# Aqui definimos o título da página e o layout como wide
st.set_page_config(page_title="Meu Dashboard", layout="wide")

#Título do seu dashboard
st.title("Meu primeiro Dashboard")
st.write("Abaixo veremos os próximos passos")

# Importar base de dados
df = pd.read_csv('NSE-TATAGLOBAL11.csv', sep=',') # Importando a base de dados e transformando em um DataFrame do pandas

# Manipulação do DataFrame
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
for i in range(0, len(data)):
    new_data.loc[i, 'Date'] = data.loc[i, 'Date']
    new_data.loc[i, 'Close'] = data.loc[i, 'Close']

# Setando o index
new_data.index = pd.to_datetime(new_data['Date'])
new_data.drop('Date', axis=1, inplace=True)
new_data.sort_index(ascending=True, inplace=True)

# Criando a base de treino e test
dataset = new_data.values
train = dataset[0:987, :]
valid = dataset[987:, :]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Criar sequências (como no LSTM)
x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(scaled_data[i - 60:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Treinar modelo MLP (rede neural simples)
model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500)
model.fit(x_train, y_train)

# Teste e predição
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i - 60:i, 0])

X_test = np.array(X_test)

closing_price = model.predict(X_test)
closing_price = closing_price.reshape(-1, 1)
closing_price = scaler.inverse_transform(closing_price)

# Dados para visualização
train_df = new_data[:987].copy()
valid_df = new_data[987:].copy()
train_df['date'] = train_df.index
valid_df['date'] = valid_df.index
valid_df['Predictions'] = closing_price

# Título do gráfico
st.subheader("Previsão de Ações")

# Gráficos com Altair
chart_train = alt.Chart(train_df).mark_line(color="blue").encode(
    x='date:T',
    y='Close:Q'
)

chart_valid = alt.Chart(valid_df).mark_line(color="green").encode(
    x='date:T',
    y='Close:Q'
)

chart_pred = alt.Chart(valid_df).mark_line(color="red").encode(
    x='date:T',
    y='Predictions:Q'
)

# Plotagem do gráfico
st.altair_chart(
    chart_train.interactive() + chart_valid.interactive() + chart_pred.interactive(),
    use_container_width=True
)