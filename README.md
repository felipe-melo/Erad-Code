# Análise de desempenho da biblioteca Theano com GPU sob a ótica do CUDA Profiler - ERAD 2017

Este trabalho está relacionado a:
1. Operações básicas de Redes Neurais;
2. Analise de desempenho do CUDA Profiler;
3. Programação de paralelo para Placas gráficas.

O [artigo](https://github.com/felipe-melo/Erad-Code/blob/master/theano_cuda.pdf) consiste em avaliar o desempenho da [biblioteca Theano](http://deeplearning.net/software/theano/index.html) em relação a realização de operações básicas de redes neurais artificiais em Placas gráficas, a ferramenta utilizada foi o [CUDA](https://www.geforce.com/hardware/technology/cuda) Profiler.
 
Dois tipos de métricas foram extraidos das execuções: O tempos de execução das operações e a quantidade de operações de ponto flutuante por segundo (flops).

7 foram as operações escolhidas para este trabalho, produto matricial, produto de hadamard, soma de matrizes, multiplação por escalar, softmax, sigmoid e tangente hiperbólica. As operações variam com respeito ao tamanho da entrada, simulando as operações matriciais que normalmente são efetuadas no treinamento de uma rede neural. Essas matrizes são compostas por números de ponto flutuando com precisão simples (32 bits).

# Execuções

Tempo:
```
$ nvprof --print-gpu-trace -u s --log-file outputFilesTimeGPU/dot/1MB_dot.out python3 __init__.py 0 512 512 10 0
```
A primeira parte, ```nvprof --print-gpu-trace -u s```, produz as métricas relativas ao tempo, em segundos, da execução de cada tarefa realizada pela placa gráfica, principalmente transferência de dados entre o host e a placa e execução das operações.

A segunda parte, ```--log-file outputFilesTimeGPU/dot/1MB_dot.out```, indica um arquivo de saída para as métricas.

a parte final do comando diz respeito a chamada código desenvolvido, o primeiro parâmetro indica qual operação, o dois seguintes dizem respeito ao tamanho da matriz que será operada, o quarto parâmetro é a quantidade de vezes que a operações será executada e o último é um booleano que representa a ativação ou não do uso de [variável compartilhada](http://deeplearning.net/software/theano/library/compile/shared.html) por parte da Theano.

Operações de ponto flutuante por segundo:
```
nvprof --metrics flop_count_sp,flop_sp_efficiency,flop_count_dp,flop_dp_efficiency --log-file outputFilesFlop/dot/1MB_dot.out python3 __init__.py 0 512 512 10 0
```
Esse comando segue os mesmos moldes da primeiro execução, a primeiro parte ativa as métricas do CUDA Profiler: flpos e cálculo de eficiência em relação as operações quanto ao uso dos recursos da GPU, jogando a saída para um arquivo especificado, a segunda parte representa os parâmetros para a execução do código, como já descrito a cima.
