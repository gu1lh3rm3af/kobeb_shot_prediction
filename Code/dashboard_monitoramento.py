from numpy.core.numeric import True_
from sklearn import metrics
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score , classification_report, f1_score, log_loss
import pycaret.classification as pc
import requests

@st.cache(allow_output_mutation=True)
def load_data(fname):
    return joblib.load(fname)

st.sidebar.title("Configurações")
st.sidebar.markdown("Selecione aqui o dataset que deseja utilizar e a métricas que deseja analisar.")

# Datasets
st.sidebar.subheader("Selecione o dataset:")
ds = st.sidebar.radio("Datasets:", ("Treino", "Operação", "Novidade"))

if ds == "Treino":
    data   = pd.read_parquet('../Data/operalization/data_train.parquet')
else:
    if ds == "Operação":
        data = pd.read_parquet('../Data/operalization/data_test.parquet')
    else:
        if ds == "Novidade":
            data = pd.read_parquet('../Data/novelty/data_novelty.parquet') 
 

rows = data.shape[0]
features = data.shape[1]

st.title("Kobe Bryant - Predição de Cestas")

st.markdown(f"""
#### Este projeto consiste em um preditor desenvilvido para tentar prever se o astro do basket Kobe Bryant acertou ou errou a cesta, analisando um dataset com registros completos dos seus arremessos, com atributos como latitude e longitute na quadra, tipo de arremesso, tempo faltante, entre outros.

#### Esta interface pode ser utilizada para a teste, avaliação e monitoramento dos resultados do modelo de classificação dos arremessos do astro Kobe Braynt. O dataset selecionado possui **{rows}** e **{features}** features. 

#### No menu ao lado é possível efetuar a configuração, definindo as métricas que deseja analisar:""")

if st.sidebar.checkbox("Exibir dados", False):
    st.subheader("Dataset")
    st.write(data)

# Variável Alvo
target_col = 'shot_made_flag'
    
# Carregar modelo

model = pc.load_model(f'./LogisticRegression')
        
# os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlruns.db'
# !mlflow models serve -m "models:/modelo_kobeb_shots_lr/Staging" --no-conda -p 5005

    # Requisição na API
host = 'localhost'
port = '5005'
url = f'http://{host}:{port}/invocations'
headers = {'Content-Type': 'application/json',}

http_data = data.drop(target_col,axis=1).to_json(orient='split')
r = requests.post(url=url, headers=headers, data=http_data)

data.loc[:, 'operation_label'] = pd.read_json(r.text).values[:,0]

# Metricas
x = data.drop(target_col,axis=1)
y = data[target_col]

def plot_metrics(metrics_list):
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x, y, display_labels= ["Acertos", "Erros"])
        st.pyplot()
    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, x, y)
        st.pyplot()
    if "Precision-Recall Curve" in metrics_list:
        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve(model, x, y)
        st.pyplot()
    if "Precision Score" in metrics_list:
        st.subheader("Precision Score")
        st.markdown(f"""#### {precision_score(data[target_col], data['operation_label'])}""")
    if "Recall Score" in metrics_list:
        st.subheader("Recall Score")
        st.markdown(f"""#### {recall_score(data[target_col], data['operation_label'])}""")
    if "F1 Score" in metrics_list:
        st.subheader("Precision Score")
        st.markdown(f"""#### {f1_score(data[target_col], data['operation_label'])}""")
    if "Log loss" in metrics_list:
        st.subheader("Log loss")
        st.markdown(f"""#### {log_loss(data[target_col], data['operation_label'])}""")
    if "Classification Report" in metrics_list:
        st.subheader("Classification Report")
        st.markdown(f"""#### {classification_report(data[target_col], data['operation_label'])}""")
        
metrics = ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Precision Score", "Classification Report", "Recall Score", "F1 Score", "Log loss"]

st.sidebar.subheader("Selecione as métricas:")
metrics_ms = st.sidebar.multiselect("Métricas", metrics)

plot_metrics(metrics_ms)

st.set_option('deprecation.showPyplotGlobalUse', False)
