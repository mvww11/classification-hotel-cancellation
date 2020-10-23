# Classificação: uma reserva de hotel será cancelada?
Esse é um projeto completo de data science: obtenção dos dados, tratamento de missing data, análise exploratória de dados, modelagem, otimização dos hiperparâmetros, explicabilidade e deploy do modelo no Google Cloud Platform.

Nessa página você encontra um resumo do projeto. A versão completa está separada nos arquivos [missing_data.ipynb](missing_data.ipynb), [EDA.ipynb](EDA.ipynb), [modeling.ipynb](modeling.ipynb), [explainability.ipynb](explainability.ipynb).

Criaremos um modelo que tentará prever se uma reserva de hotel será cancelada com base em cerca de 60 informações disponíveis sobre a reserva, como número de adultos, quantidade de diárias e tempo de estadia.

**Se o hotel souber com antecedência quais são as reservas que têm alta probabilidade de serem canceladas, ele pode tomar medidas de marketing para evitar esse cancelamento (oferecendo alguma vantagem extra, por exemplo). Como cerca de 41% das reservas são canceladas, o projeto tem um grande potencial de retorno para o negócio.**

## Resumo do Projeto
* Objetivo: criar um modelo de previsão da probabilidade de uma reserva de hotel ser cancelada.
* Nosso modelo xgboost final alcançou um recall de 92% em data points nunca vistos por ele.
* Dados: 80 mil reservas de um hotel situado em Lisboa, Portugal.
* Análise exploratória de dados mostrou que a renda é o fator mais relevante para a previsão da nota.
* Feature engineering: criei duas features novas: uma que indica a renda per capita (por residente no domicílio) do candidato e outra que indica a escolaridade máxima entre pai e mãe.
* Benchmark model com XGBoost e LightGBM para análise de importâncias relativas entre features e feature selection.
* Refinamento do modelo: procura por hiperparâmetros ótimos usando bayesian search.
* Interpretação do modelo: expliquei quais são as decisões que o modelo faz para chegar a uma previsão. Para isso, usei valores SHAP.
* Deploy serverless do modelo no [AWS Lambda](https://aws.amazon.com/lambda/) e criação de um [bot do Telegram](https://telegram.org/blog/bot-revolution) que permite que qualquer pessoa faça a previsão da sua nota no ENEM usando nosso modelo.

## Recursos utilizados
**Python**: Versão 3.7<br>
**Pacotes Python**: numpy, pandas, matplotlib, seaborn, xgboost, hyperopt, joblib, shap<br>
**Serverless framework para deploy no AWS Lambda**: https://www.serverless.com/<br>
**Bayesian optimization**: [[1]](https://towardsdatascience.com/automated-machine-learning-hyperparameter-tuning-in-python-dfda59b72f8a) [[2]](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-ex.html)<br>
**Explicando o modelo com SHAP**: [[1]](https://medium.com/@gabrieltseng/interpreting-complex-models-with-shap-values-1c187db6ec83) [[2]](https://towardsdatascience.com/shap-explained-the-way-i-wish-someone-explained-it-to-me-ab81cc69ef30) [[3]](https://towardsdatascience.com/black-box-models-are-actually-more-explainable-than-a-logistic-regression-f263c22795d) [[4]](https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d)

## Obtenção dos dados
Os dados foram disponibilizados no artigo [Hotel booking demand datasets](https://www.sciencedirect.com/science/article/pii/S2352340918315191) e coletados [aqui](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-02-11). São cerca de 80 mil reservas feitas num hotel situado na cidade de Lisboa, Portugal, entre os anos de 2015 e 2017.

Exemplos de features disponíveis:
|variable                       |class     |description |
|:------------------------------|:---------|:-----------|
|is_canceled                    |double    | Value indicating if the booking was canceled (1) or not (0) |
|lead_time                      |double    | Number of days that elapsed between the entering date of the booking into the PMS and the arrival date |
|adults                         |double    | Number of adults|
|children                       |double    | Number of children|
|country                        |character | Country of origin. Categories are represented in the ISO 3155–3:2013 format |
|is_repeated_guest              |double    | Value indicating if the booking name was from a repeated guest (1) or not (0) |
|reserved_room_type             |character | Code of room type reserved. Code is presented instead of designation for anonymity reasons |
|adr                            |double    | Average Daily Rate as defined by dividing the sum of all lodging transactions by the total number of staying nights |
|total_of_special_requests      |double    | Number of special requests made by the customer (e.g. twin bed or high floor)|

fonte: adaptado do [repo](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-02-11).

A variável in_canceled informa se a reserva foi cancelada (in_canceled = 1) ou não (in_canceled = 0). Essa é a variável dependente, aquela que queremos que nosso modelo preveja.

## Data Cleaning (tratando missing data)
Após carregar os dados, precisei fazer uma série de transformações para que ficassem apropriados para serem utilizados no treinamento dos modelos. Confira a etapa completa em [missing_data.ipynb](missing_data.ipynb).
* Removi cerca de 30 data points continham campos nulos na coluna Country.
* Removi a coluna Company, que possuía mais de 90% de missing data.
* Removi 324 reservas que possuíam duração de hospedagem de 0 dias.
* Removi 99 reservas que tinham 0 pessoas associadas (nenhum adulto, criança ou bebê).
* Transformei o Data type de features categóricas de string para número inteiro.

## Análise Exploratória de Dados e Feature Engineering
Após o tratamento de missing data, ficamos com 78879 data points. Entre essas reservas, 41% foram canceladas. Isso indica que o cancelamento de reservas tem um impacto muito grande no business. Se conseguirmos diminuir esse percentual, o potencial de geração de lucro para o negócio é enorme.

Abaixo estão ilustrados alguns insights observados na Análise Exploratória, e as Feature Engineering realizadas. A análise completa está no arquivo [EDA.ipynb](EDA.ipynb).
* A proporção de cancelamentos era maior em reservas feitas por clientes de Portugal.
* A proporção de cancelamentos era menor em reservas feitas por clientes da União Europeia que não de Portugal.
* As duas informações acima me levaram a fazer 2 Feature Engineering: **isPRT**: a reserva foi feita por um cliente de Portugal? **isEU**: a reserva foi feita por um cliente da união Europeia? Confira o gráfico abaixo.
* 40% das reservas possuíam algum tipo de pedido especial, e tinham uma taxa de cancelamento 2.5x menor que reservas sem nenhum pedido especial.
* Reservas que possuíam apenas dias de final de semana tinham uma taxa de cancelamento menor
* A informação acima me levou a criar a seguinte feature: **isOnlyWeekend**: a reserva possui apenas dias de final de semana?
<img src='isPRT_cancel.png' width="400">



## Data Leakage
Algumas features de nosso data set foram eliminadas antes do treinamento do modelo, para evitar data leakage. Por exemplo, a coluna 'ReservationStatus' ( que possui 3 valores possíveis: 'cancelled', 'no-show', 'check-out') determina completamente se a reserva foi cancelada ou não. Entretanto, quando o modelo for colocado em produção, ele tentará prever se a reserva será cancelada ANTES de termos a informação sobre o 'ReservationStatus'. Por isso, nosso modelo não pode usar essa informação no treinamento. O mesmo vale para a coluna 'ReservationStatusDate' e para a coluna 'AssignedRoomType'. Logo, essas 3 colunas foram eliminadas da análise.

## Modelagem e split dos dados
Usaremos um modelo de Gradient Boosting com a implementação da biblioteca xgboost. O processo completo de modelagem pode ser visto no arquivo [modeling.ipynb](modeling.ipynb).

Faremos um split de 60/20/20% dos dados em conjuntos de treinamento, validação e teste, respectivamente. O conjunto de treinamento será aquele em que ajustaremos os parâmetros treináveis de nosso modelo. Como treinaremos vários modelos, que diferem por seus hiperparâmetros, usaremos o conjunto de validação avaliar qual é o melhor entre eles. Já o conjunto de teste será usado uma única vez no final do projeto para estimar a performance que o modelo terá em produção. Desse modo, o conjunto de teste será composto de pontos nunca vistos pelo modelo durante seu treinamento e refinamento.

A métrica que usaremos para avaliar qual é o melhor modelo será a área sob a curva ROC, conhecida como AUC (area under curve). Quanto maior o valor dessa métrica, melhor é o trade-off que teremos entre positivos verdadeiros e positivos falsos (i.e., entre o modelo acertar quais reservas serão canceladas e não errar as reservas não seriam canceladas).

## Benchmark
Um modelo inicial foi treinado com xgboost, usando os hiperparâmetros default. Para não precisar tratar o número de árvores no modelo como um hiperparâmetro a ser otimizado, definimos que interromperíamos o treinamento quando nossa métrica de auc não apresentasse melhora por 30 árvores seguidas (early_stopping_rounds).

Nosso modelo de benchmark alcançou um AUC de 0.946 e um Recall de 82.54%. Ou seja, o modelo previu corretamente 82.54% das reservas que seriam canceladas.

## Refinamento do modelo
O processo de refinamento consiste em treinar diferentes modelos, que diferem pelos seus hiperparâmetros, e utilizar nosso conjunto de validação para verificar qual dos modelos faz a melhor previsão. Compararemos a qualidade dos modelos pela métrica do AUC.

Para buscar os melhores hiperparâmetros, utilizaremos o [Bayesian optimization](https://towardsdatascience.com/automated-machine-learning-hyperparameter-tuning-in-python-dfda59b72f8a). Nesse método, sorteamos aleatoriamente valores para os hiperparâmetros de acordo com uma distribuição de probabilidade. O Bayesian optimization utiliza iterativamente os resultados que vão sendo obtidos para explorar mais intensamente os intervalos de valores mais promissores, para cada hiperparâmetro. Assim, a cada novo modelo treinado, é atualizada a distribuição de probabilidade dos valores associados a cada hiperparâmetro. A implementação do Bayesian optimization que utilizaremos aqui é a do pacote [hyperopt](https://github.com/hyperopt/hyperopt).

Como o pacote [hyperopt](https://github.com/hyperopt/hyperopt) precisa de uma função para MINIMIZAR, vamos definí-la como 1 - AUC. Assim, minimizando o AUC, estamos maximizando o AUC.

Após treinarmos 320 modelos diferentes, o melhor entre eles apresentou um AUC de 0.952 e um Recall de 83.42%.

O processo completo de refinamento também pode ser visto no arquivo [modeling.ipynb](modeling.ipynb).

## Avaliando Overfitting
nosso modelo alcançou um f1_score de 96.62% no conjunto de treinamento, contra 85.26% no conjunto de validação. Isso aponta que nosso modelo tem um certo grau de overfitting, uma vez que sua performance é consideravelmente melhor nos pontos em que foi treinado.

Para minimizar o overfitting podemos otimizar certos hiperparâmetros do XGBoosting com que ainda não trabalhamos, como o subsample. Outra estratégia para combater o problema é aumentar o tamanho de nosso conjunto de treinamento. Para isso, uma possibilidade é fazermos k-fold validation, ao invés de usar um conjunto separado para validação.

## Otimizando para o Recall
Acredito que o recall tenha uma relevância grande para o projeto. Isso porque o recall mede o percentual de acertos do modelo nas reservas que são canceladas, de fato. E se o hotel sabe de antemão que uma reserva tem alta probabilidade de cancelamento, pode agir para evitar esse cancelamento (oferecendo algum benefício ao cliente, por exemplo). Isso tem um grande potencial de impacto no negócio, uma vez que, como vimos, 41% de todas as reservas são canceladas, em média.

Até agora, para que nosso modelo previsse que uma reserva seria cancelada, era necessário que a probabilidade de cancelamento calculada por ele para essa reserva chegasse a 50%. Podemos otimizar o recall diminuindo esse threshold de probabilidade. Assim, por exemplo, uma reserva com 45% de probabilidade de cancelamento já seria prevista como cancelada por nosso modelo.

Temos um trade-off, no entanto. Ao diminuir esse threshold, estaremos cometendo, com mais frequência, o erro de classificar como canceladas reservas que, de fato, não seriam canceladas. O impacto desse erro para o negócio é que o hotel oferecerá, com maior frequência, algum tipo de vantagem para clientes que não iriam cancelar suas reservas.

Para avaliar o trade-off e escolhermos o melhor threshold, plotaremos abaixo o precision, recall e f1-score em função do threshold para o conjunto de validação.
<img src='threshold.png' width="400">
Como previsto, à medida que o recall aumenta, a precision diminui. Como estamos dando mais importância ao recall, conforme explicado acima, vou escolher o threshold de 0.25. Em outras palavras, toda vez que nosso modelo calcular que a probabilidade de uma reserva ser cancelada é maior que 25%, nossa previsão será de que aquela reserva será cancelada.

O threshold de 25% dá, no conjunto de validação, um recall de 92.12% e uma precision de 77.33%.

Isso significa que, em nosso conjunto de validação, sempre que uma reserva vai ser cancelada pelo cliente, nosso modelo consegue prever corretamente em 92.12% dos casos. Em contrapartida, de todas as vezes que nosso modelo diz que uma reserva será cancelada, ele acerta em 77.33% das vezes. Ou seja, em 22.66% das vezes que nosso modelo diz que uma reserva será cancelada, na verdade ela não é.

## Estimando a acurácia do modelo em produção
Para estimar o desempenho do nosso modelo em produção, vamos utilizar o conjunto de teste. Perceba que essa é a primeira vez que utilizamos o conjunto de teste na modelagem. Isso garante que nenhum parâmetro ou hiperparâmetro foi selecionado para otimizar o modelo para esse conjunto.

Os pontos que temos no conjunto de teste nunca foram vistos pelo modelo, de modo que seu desempenho nesse conjunto é uma medida mais acurada do desempenho que o modelo terá no mundo real, em produção.

O modelo teve um desempenho no conjunto de teste muito próximo daquele observado no conjunto de validação (recall de 92,02% no teste, contra 92,12% na validação). Isso significa que nosso modelo generaliza bem para pontos inéditos, e nos dá maior confiança de usá-lo em produção.

A estimativa final é que nosso consegue prever corretamente 92.02% das reservas que serão canceladas. Como o precision encontrado foi de 77.44%, então em 22.56% das vezes que nosso modelo diz que uma reserva será cancelada, ele erra.

## Interpretação do modelo
Para explicar quais são as decisões que o modelo toma para chegar às previsões, utilizei os valores SHAP, com a implementação da biblioteca [shap](https://github.com/slundberg/shap). Seguem alguns dos insights percebidos.

#### Impacto das features na previsão do modelo
O gráfico abaixo dá uma visão geral das decisões que nosso modelo faz para chegar à previsão de um usuário. Cada linha representa uma feature diferente e deve ser lida separadamente.

<img src='imgs/shap-summary2.png'>

A primeira linha, da renda (Q006), mostra que, dentro dos candidatos das menores classes de renda (em azul), temos dois efeitos distintos. A maior parte deles tem um grande prejuízo na previsão da nota por conta de sua classe de renda (cluster centrado no valor de SHAP de -30). Mas existe outro cluster, de menor tamanho, centrado perto de SHAP zero. Para esse segundo cluster, o fator renda não tem grande impacto na previsão da nota. É provável que seja contrabalenceado por outro fator que não incluso no modelo (possivelmente alguma coisa relacionada à motivação ou dedicação que o aluno dá aos estudos, devido à diferenças culturais ou de estímulos familiares, fator que não é contemplado por nosso modelo).

A segunda linha mostra que quando o candidato é mulher (vermelho, pois TP_SEXO = 1 para mulheres) a previsão de nota é menor do que para homens. Essa queda na previsão da nota devida ao fator "sexo" é bastante consistente entre os candidatos, como podemos observar pelos dois cluster de pequena variância.

A terceira linha aponta que quando a escolaridade dos pais é baixa (em azul), o prejuízo na previsão da nota prevista por nosso modelo é variado. Há alunos com relativamente pouco prejuízo (SHAP pouco negativo) e outros com muito prejuízo (SHAP muito negativo). Já para alta escolaridade, o benefício na nota é mais consistente (cluster com pequena variância).

#### O efeito do sexo sobre a previsão do modelo, e sua interação com a renda
O gráfico abaixo mostra que quando o candidato é mulher (TP_SEXO = 1), o valor de SHAP é negativo (entre -10 e -25, aproximadamente). Como já havíamos percebido, nosso modelo dá uma previsão de nota menor quando o candidato é mulher.
<img src='imgs/shap-dependence-sexo.png'>

Podemos, contudo, observar um efeito curioso. A queda da nota devido ao fator "sexo" para mulheres é mais acentuada quando as candidatas são das maiores classes de renda: os pontos de cor vermelha (alta renda), para as mulheres, estão mais abaixo do que os de cor azulada (baixa renda). Ou seja, o "prejuízo" na nota causado por ser mulher é mitigado quando a candidata pertence às menores classes de renda, e reforçado quando pertence às maiores classes de renda.

Já para os homens, o oposto é verdadeiro. O aumento na nota prevista por nosso modelo que tem como fundamento o fator "sexo" é reforçado quando o candidato pertence às maiores classes de renda, e diminuído quando pertence às menores classes de renda.

#### O efeito da renda sobre a previsão do modelo, e sua interação com a escolaridade dos pais
Confirmamos, pelo gráfico a seguir, que uma maior classe de renda faz com que nosso modelo preveja notas maiores. Entretanto, um efeito que fica evidente é que, principalmente para classes de renda intermediárias, o aumento na nota devido a uma maior renda é diminuído caso a escolaridade dos pais seja baixa.

<img src='imgs/shap-dependence-renda.png'>

## Colocando o modelo em produção
Optamos por fazer um deploy serverless do modelo no AWS Lambda. Isso porque os requests ao API end-point seriam esporádicos, de modo que não precisamos de uma máquina continuamente dedicada para processá-los.

Utilizamos o serverless framework para colocar o modelo em produção. Assim, a partir de uma url (API end-point), podemos enviar um request contendo os inputs das features (renda, número de residentes, sexo e max_escol) e receber de volta o valor de previsão da nota do candidato.

Por exemplo, a seguinte URL receberá (na variável 'predictedGrade') a previsão de 510.06 pontos na nota do ENEM para um candidato de features dadas por {'Q005': 1, 'Q006': 3, 'TP_SEXO':1, 'max_escol': 4, 'perCapita': 6}:
https://ojlzl0q4wg.execute-api.sa-east-1.amazonaws.com/dev/get-grade?Q005=1&Q006=3&TP_SEXO=1&max_escol=4&perCapita=6

Criei também uma interface por meio de um bot no Telegram para que qualquer pessoa pudesse fazer a previsão da sua nota usando o modelo desenvolvido aqui. Esse bot utiliza a URL acima para fazer os requests das previsões, passando os dados digitados pelo usuário. Abaixo estão mostrados print screens do bot em funcionamento.

<img src="imgs/bot1.jpeg" width="250">
<img src="imgs/bot2.jpeg" width="250">
<img src="imgs/bot6.jpeg" width="250">
<img src="imgs/bot3.jpeg" width="250"> 
<img src="imgs/bot4.jpeg" width="250">
