# Kobe Bryant - Análise Preditiva de Acertos
------------
O objetivo deste projeto é contruir um preditor capaz de prever se o astro do basket Kobe Bryant acertou ou errou a cesta, analisando um dataset com registros completos dos seus arremessos, com atributos como latitude e longitute na quadra, tipo de arremesso, tempo faltante, entre outros.

Link para o [dataset](https://www.kaggle.com/code/jeongwonkim10516/kobe-bryant-shot-selection-increase-prediction/data").

Para este experimento, utilizaremos duas abordagens, regressão logística e árvore de decisão, e demonstraremos o processo completo de um projeto de machine learning, desde o coleta e preparação dos dados, passando pelo pré-processamento, treinamento e avaliação, monitoramento, atualização e deploymento.

O projeto seguirá o framework TDSP (Team Data Science Process) da Microsoft. Um framework ágil com metodologia de data science interativa proposto para entregar soluções de ML de forma eficiente (MLOps). O foco do TDSP é garantir a colaboração e a otimização do trabalho entre pessoas/times através de melhores práticas de implementação e infraestrutura levantadas pela Microsoft junto com grandes players do mercado no cenário de aplicações inteligentes e machine learning.

A ilustração abaixo demostra o ciclo de vida de uma aplicação de data science:

![](https://docs.microsoft.com/en-us/azure/architecture/data-science-process/media/overview/tdsp-lifecycle2.png)

Os principais papéis envolvidos num projeto de data science, segundo o TDPS são: 

* Solution architect
* Project manager
* Data engineer
* Data scientist
* Application developer
* Project lead

O ilustração a seguir descreve as principais responsabilidades de cada papel dentro de um projeto de DS. Importante ressaltar que este projeto demonstrará as responsabilidades do Data scientist:

![](https://docs.microsoft.com/en-us/azure/architecture/data-science-process/media/overview/tdsp-tasks-by-roles.png#lightbox)

Para desenvolvimento do modelo utilizaremos como principal biblioteca o pyCaret,  uma biblioteca de AutoML do Python que permite o desenvolvimento de todo o ciclo da criação de um modelo de Machine Learning com poucas linhas de código.

Para rastreamento e monitoramento do modelo utilizaremos o MLFlow em conjunto com o streamLit. O MLFlow é uma biblioteca open-source para gerenciar o ciclo de vida dos experimentos, e o streamlit, é um framework também open-source, cujo principal objetivo é disponibilizar o projeto em produção, através de dashboards interativos, sem a necessidade de conhecer ferramentas de front-end ou de deploy de aplicações.

Com o MLFlow, faremos o controle do nosso modelo entre os pipelines de desenvolvimento e producão. A implementação de pipelines distintos é importante e necessária, pois podem haver diversas diferenças entre os dois cenários, como:

* Ausência da resposta correta para avaliarmos o modelo no ambiente de desenvolvimento, pois utilizamos muitas vezes bases offline com fotografias até determinada data, e os dados de produção serão os mais atuais possível.
* Pipelines distintos de coleta de dados entre ambiente de dev e produção, onde muitas vezes em produção necessitaremos do complemento de algumas informações.
* Diferentes condições de operação (frequência de resposta, latência…)
* Apresentação do resultado para usuário final de forma a disponibilizar insumos para análises e tomadas de decisão
* Retreinamento dos modelos em produção para recalibração dos parâmetro.
* Segurança nas manutenções, mitigando o risco de quebra da aplicação em produção.
