# Classificação de Dígitos (MNIST) com Redes Neurais Profundas

Este repositório contém uma implementação de uma Rede Neural Multicamadas (MLP - Multilayer Perceptron) utilizando a biblioteca **PyTorch** para a classificação de dígitos manuscritos do dataset **MNIST**.

O projeto foi desenvolvido como parte de estudos independentes sobre Machine Learning, com foco em entender o fluxo de treinamento, retropropagação (backpropagation) e manipulação de tensores.

## Tecnologias Utilizadas

* **Python 3**
* **PyTorch**: Framework principal para construção da rede e autograd.
* **Torchvision**: Utilizado para carregamento e transformação do dataset MNIST.

## Arquitetura do Modelo

A rede neural implementada possui a seguinte estrutura:

1.  **Camada de Entrada**: 784 neurônios (correspondentes aos pixels 28x28 das imagens).
2.  **Camadas Escondidas**:
    * Camada Linear 1: 256 neurônios + ativação ReLU.
    * Camada Linear 2: 128 neurônios + ativação ReLU.
    * Camada Linear 3: 64 neurônios + ativação ReLU.
3.  **Camada de Saída**: 10 neurônios (um para cada classe de 0 a 9).

### Justificativas Técnicas:
* **Otimizador Adam**: Escolhido pela sua taxa de aprendizado adaptativa, acelerando a convergência.
* **CrossEntropyLoss**: Critério padrão para problemas de classificação multiclasse.
* **Modularização**: O código utiliza funções distintas para treino e avaliação.

## 📊 Resultados

Após o treinamento de apenas **1 época**, o modelo atingiu os seguintes resultados:

* **Acurácia no Treino**: 57476/60000 (95.79%)
* **Acurácia no Teste**: 9552/10000 (95.52%)

## Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/navxmane/simple-neural-network-pytorch.git
```
