# RaNN: Randomized Neural Network Autoencoder para Análise de Complexidade Estrutural

Este repositório contém a implementação de um modelo de Autoencoder fundamentado em Redes Neurais Aleatórias (RaNN - *Randomized Neural Networks*), desenvolvido nativamente utilizando a biblioteca `numpy`. O objetivo primordial deste modelo é o processamento espacial de imagens, promovendo a compressão da representação da entrada através de um estrangulamento de dimensão (*bottleneck*) e a subsequente reconstrução do sinal. Tal processo permite o mapeamento topográfico das áreas de maior complexidade estrutural, identificadas por regiões onde a rede apresenta maior magnitude de resíduo na reconstrução.

## 1. Fundamentação Teórica da RaNN

Em contraste com os métodos de otimização estocástica baseados em gradiente (tais como o *Backpropagation* empregado em redes neurais profundas), o modelo RaNN (frequentemente discutido na literatura sob as nomenclaturas de *Random Vector Functional Link* (RVFL) ou *Extreme Learning Machines* (ELM)) propõe uma formulação analítica para o treinamento, apresentando notável eficiência computacional. A arquitetura opera sob os seguintes axiomas:

1. **Projeção Aleatória do Espaço de Entrada**: Os pesos sinápticos (`W_in`) e vieses (`b_in`) da camada oculta são amostrados de uma distribuição aleatória e mantidos estáticos durante todo o procedimento de treinamento.
2. **Mapeamento Não-Linear**: Os dados de entrada são projetados em um espaço de características de dimensionalidade ajustável por intermédio de uma função de ativação não-linear (neste escopo, adotou-se a tangente hiperbólica).
3. **Treinamento Analítico e Convexo**: Exclusivamente os pesos da camada de saída (`W_out`) são otimizados. A determinação desta matriz de pesos configura-se como um problema linear de mínimos quadrados, resolvido instantaneamente de forma exata (fechada) via Regressão Ridge (utilizando a Inversa Generalizada de Moore-Penrose associada a um fator de regularização de Tikhonov).

### 1.1 O Papel do Autoencoder e do Gargalo de Informação

No contexto deste projeto, a RaNN é estruturada como um Autoencoder de Compressão. A imagem de entrada é segmentada em sub-matrizes (*patches*) de dimensões $8 \times 8$ pixels. A camada oculta foi deliberadamente parametrizada para possuir apenas 10% da dimensionalidade intrínseca dos dados de entrada. 

Este gargalo estrutural força o modelo a convergir para uma projeção que retenha apenas os componentes mais invariantes e representativos da imagem, suprimindo o ruído. Consequentemente, quando o processo de decodificação tenta reconstruir um *patch* de alta densidade informacional (e.g., com texturas complexas ou arestas de alta frequência), a magnitude do erro residual eleva-se de maneira proporcional à complexidade local.

## 2. Metodologia Computacional e Pipeline

A execução do modelo em `rann.py` obedece à seguinte sequência algorítmica:

1. **Pré-Processamento e Interpolação**: A imagem `input.jpg` é carregada no ambiente e interpolada para que suas dimensões consistam em múltiplos exatos do tamanho do *patch* definido.
2. **Particionamento Espacial (Patching)**: A matriz bidimensional da imagem é fracionada em blocos disjuntos de $8 \times 8$ pixels. Tais blocos sofrem um processo de achatamento vetorial (conversão para 1D) para ingressar na topologia da rede.
3. **Inferência Analítica e Reconstrução**: Os vetores são projetados no subespaço latente (reduzido a 10% da dimensão original). Em seguida, computa-se a matriz ótima de decodificação para a minimização do erro de reconstrução.
4. **Cômputo do Resíduo Absoluto**: Determina-se a divergência absoluta, normalizada entre os canais de cor, entre o sinal de entrada original e o sinal sintetizado pelo Autoencoder.
5. **Geração dos Mapas de Representação**:
   - É computado um mapa em tons de cinza denotado por **Resíduo Absoluto**, retratando a magnitude estrita do erro por pixel.
   - Um mapa visual utilizando mesclagem alfa (*Alpha Blending*) é sintetizado sobre o sinal de entrada, denotado por **Mapa de Complexidade Estrutural**. A intensidade cromática (azulada) deste mapa atua como função direta da complexidade da textura e da topologia geométrica da região avaliada.

## 3. Instruções de Execução

### 3.1 Dependências de Software
A execução demanda pacotes padronizados da linguagem Python orientados a cálculo numérico e processamento de matrizes:
```bash
pip install numpy opencv-python matplotlib
```

### 3.2 Execução do Módulo Principal
O diretório operacional deve conter a imagem de avaliação sob a nomenclatura `input.jpg`. A execução dá-se via terminal:
```bash
python rann.py
```

O algoritmo gerará, ao término do processamento, o arquivo rasterizado `resultado.jpg` contendo a composição analítica estruturada em uma grade $2 \times 2$:
1. **Entrada (Ground Truth)**
2. **Sinal Reconstruído** (Saída da RaNN)
3. **Resíduo Absoluto**
4. **Mapa de Complexidade Estrutural**

## 4. Referências Bibliográficas

Os fundamentos matemáticos referentes ao treinamento de redes neurais com conexões e pesos ocultos pseudo-aleatórios derivam da literatura seminal consolidada nas seguintes publicações:

- **Schmidt, W. F., Kraaijveld, M. A., & Duin, R. P. W. (1992)**. *Feedforward neural networks with random weights*. In 11th IAPR International Conference on Pattern Recognition. Vol. II. Conference B: Pattern Recognition Methodology and Systems (pp. 1-4). IEEE.
- **Pao, Y. H., Park, G. H., & Sobajic, D. J. (1994)**. *Learning and generalization characteristics of the random vector functional-link net*. Neurocomputing, 6(2), 163-180.
- **Huang, G. B., Zhu, Q. Y., & Siew, C. K. (2006)**. *Extreme learning machine: theory and applications*. Neurocomputing, 70(1-3), 489-501. 
