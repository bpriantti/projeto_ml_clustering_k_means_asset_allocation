# Projeto ML clustering - k-means asset allocation

__Bussines Problem:__
> Durante a rotina de investimentos torna-se importante para a construção de portfólios de ativos, a seleção de ativos para compor uma carteira de investimentos, sob essa ótica torna-se necessário encontrar características em comum entre os ativos para a composição de bons portfólios.

__Objetivo:__   
> Desenvolver um modelo Machine Learning para clustering de ativos base utilizando as métricas CAGR x Anualizated Vol, para as ações listadas no índice IBRX100.

__Autor:__  
   - Bruno Priantti.
    
__Contato:__  
  - bpriantti@gmail.com

__Encontre-me:__  
   -  https://www.linkedin.com/in/bpriantti/  
   -  https://github.com/bpriantti
   -  https://www.instagram.com/brunopriantti/
   
__Frameworks Utilizados:__

- Numpy: https://numpy.org/doc/  
- Pandas: https://pandas.pydata.org/
- Matplotlib: https://matplotlib.org/ 
- Seaborn: https://seaborn.pydata.org/  
- Plotly: https://plotly.com/  
- Scikit learn: https://scikit-learn.org/stable/index.html
- Statsmodels: https://www.statsmodels.org/stable/index.html

__Project Steps:__

__Step 01:__ Processo de Aquisição de Dados:

> Realizou-se o processo de ETL com a API de dados, yfinance e com isso obteve-se a série histórica dos ativos listados no índice IBRX100, abaixo vemos a lista de tickers:

> Asset List:

       ['WEGE3', 'EMBR3', 'AZUL4', 'CCRO3', 'ECOR3', 'GOLL4',
       'RAIL3', 'STBP3', 'POSI3', 'BRFS3', 'JBSS3', 'MRFG3',
       'BEEF3', 'SMTO3', 'ABEV3', 'ASAI3', 'CRFB3', 'NTCO3',
       'RAIZ4', 'SLCE3', 'AMER3', 'ARZZ3', 'SOMA3', 'AMAR3',
       'LREN3', 'MGLU3', 'PETZ3', 'VIIA3', 'ALPA4', 'CYRE3',
       'EZTC3', 'MRVE3', 'CVCB3', 'COGN3', 'RENT3', 'MOVI3',
       'YDUQ3', 'BRML3', 'IGTI11', 'MULT3', 'BPAN4', 'BBDC3',
       'BBDC4', 'BBAS3', 'BPAC11', 'ITSA4', 'ITUB4','SANB11',
       'WIZS3', 'B3SA3', 'CIEL3', 'DXCO3', 'KLBN11', 'SUZB3',
       'BRAP4', 'CMIN3', 'VALE3', 'BRKM5', 'GGBR4', 'GOAU4',
       'CSNA3', 'USIM5', 'RRRP3', 'CSAN3', 'PETR3', 'PETR4',
       'PRIO3', 'UGPA3', 'VBBR3', 'HYPE3', 'RADL3', 'FLRY3',
       'HAPV3', 'QUAL3', 'RDOR3', 'LWSA3', 'CASH3', 'TOTS3',
       'VIVT3', 'TIMS3', 'SBSP3', 'AURE3', 'CMIG4', 'CPLE6',
       'CPFE3', 'ELET3', 'ELET6', 'ENBR3', 'ENGI11', 'ENEV3',
       'EGIE3', 'EQTL3', 'TAEE11','BBSE3', 'IRBR3', 'PSSA3', 
       'SULA11',
       
- Obs: foram removidos os ativos, [PCAR3,SUZB3], devido a problemas de market data.

__Step 02:__ Processo de Data Visualization:

> Utilizando a lib plotly, verificou-se os dados e sua série de de fechamentos ajustados histórica, como demostrado na imagem abaixo:

<img src="https://github.com/bpriantti/Projeto_ML_clustering_k_means_asset_allocation/blob/main/images/image-1.png?raw=true"  width="600">

__Step 02:__ Calculo Features:

> Como o objetivo é analisar os ativos em uma ótica de retorno e volatilidade realizou-se o cálculo das features, CAGR(Compound Annual Growth Rate) e volatilidade anualizada, abaixo visualizamos o scatter plot das features para toda a nossa data base:

<img src="https://github.com/bpriantti/Projeto_ML_clustering_k_means_asset_allocation/blob/main/images/image-2.png?raw=true"  width="600">

__Step 03:__ ML k-means research:

> Para o set inicial do número de clusters utilizamos o Elbon Method, que consiste na escolha ótima do número de clusters na quebra da curvatura da curva de clusters, o valor escolhido para o número de k-means, foi 5.

<img src="https://github.com/bpriantti/Projeto_ML_clustering_k_means_asset_allocation/blob/main/images/image-3.png?raw=true"  width="600">

__Step 04:__ ML k-means train:

> Apos o numero de clusters ter sido selecionado, realizou-se o treinamento do k-means e em seguida realizou-se o plot dos dados agora com o numero de k-means ajustado pelas cores.

> data.head(): 

|index  |Asset   |CAGR   |Annualised_Vol   |Clusters   |
|---|---|---|---|---|
|0   |WEGE3   |31.09    |0.39   |4   |
|1   |EMBR3   |-37.07   |0.53   |1   |
|2   |AZUL4   |-44.16   |0.70   |1   |
|3   |CCRO3   |-18.58   |0.44   |3   |
|4   |ECOR3   |-32.76   |0.47   |3   |

- k-means plot:

<img src="https://github.com/bpriantti/Projeto_ML_clustering_k_means_asset_allocation/blob/main/images/image-4.png?raw=true"  width="800">

__Step 05:__ Construindo Portifolio com ativos do Cluster - 04:

Foram selecionados ativos do Cluster 04, este que após análise de risco retorno possui melhor equilíbrio entre risco e retorno sobre o investimento.

<img src="https://github.com/bpriantti/Projeto_ML_clustering_k_means_asset_allocation/blob/main/images/image-5.png?raw=true"  width="800">

Como observados o portfólio desenvolvido supera o índice ibovespa no período histórico de dados, agora vamos analisar algumas métricas de risco vs retorno:

|item  |Portifolio   |Benchmark|
|---|---|---|
|Retorno(%):   |305.6%  |120.8%   |
|Max Drawdown  |-32.71  |-50.24   |

Como podemos observar o portfólio desenvolvido consegue superar o benchmark correndo um risco menor que o mesmo.
