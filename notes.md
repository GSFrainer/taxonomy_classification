# A geração e padronização dos conjuntos (Treino/Teste) de dados 
- Arquivo: ./data_generator.ipynb
- Foi criado o script Data_Generator
    - O script:
        1- Carrega o dataset do PR2 pré-filtrado (cleaned_sequences) completo
        2- Para cada nível taxômico:
            2.1- Tratamento e Filtragem dos dados
                2.1.1- Seleciona os registros com valores não nulos no nível
                2.1.2- Seleciona as sequências com o tamanho correto (900 bases)
                    -- Filtrar por poucas bases dificultam o processo de ML
                    -- Filtrar por muitas bases faz com que sobrem poucos registros
                2.1.3- Remove os registros em que uma mesma sequência é classificada em dois ou mais grupos diferentes
                    -- Sequencias iguais em grupos diferentes não servem como bons identificadores e podem atrapalhar no processo de identificação e abstração de padrões
                2.1.4- Remove registros de categorias que tiverem menos de 10 exemplares
                    -- Categorias com poucos exempleres podem gerar ruídos no aprendizado
                    -- Aumentam o desbalanceamento do dataset
                    -- Chances significativas de serem selecionados apenas para o conjunto de treino ou apenas pro conjunto de teste
                2.1.5- Atribuí valor nulo para todos os níveis subsequentes, mantendo apenas as classificações nos anteriores (superiores) e no atual
                2.1.6- Remove todos os registros que tenham valor nulo na classificação em algum nível anterior
            2.2- Divisão de Treino e Teste
                2.2.1- São aplicadas a funções de amostragem dos dados
                    2.2.1.1 - Para amostragem aleatória, são selecionados 10% dos registros para compôrem o conjunto de Teste
                    2.2.1.2 - Para amostragem estratificada, são separados 10 grupos mantendo a proporção das classes. Um dos grupos torna-se o conjunto de Teste
                2.2.2- Os registros restantes são selecionados para o conjunto de Treino
            2.3- São salvos os arquivos csv referentes aos conjuntos de treino e teste, separadamente
            2.4- Para cada um dos conjuntos, é gerado o arquivo de taxonomia e um arquivo fasta, compatíveis com o plug-in feature-classifier do QIIME2
                2.4.1- Para gerar os arquivos de taxonomia de cada conjunto:
                    2.4.1.1- Em cada registro do conjunto, é aplicada uma função para formatar de acordo com a estrutura usada pelo plug-in
                    2.4.1.2- Os dados formatados são salvos junto com o id da sequência em um arquivo de taxonomia
                        -- Esse arquivo é salvo na pasta do respectivo nível
                        -- O nome segue o padrão "PR2_train_taxonomy.txt" para os referentes ao conjunto de treino e "PR2_test_taxonomy.txt" para os referentes ao conjunto de teste
                2.4.2- Para gerar os arquivos fasta de cada conjunto:
                    2.4.2.1- É criado um arquivo com extensão fasta
                        -- Esse arquivo é salvo na pasta do respectivo nível
                        -- Os nomes dos arquivos fasta do conjunto de treino e de teste, respectivamente são "PR2_train.fasta" e "PR2_test.fasta"
                    2.4.2.2- Para cada registro no conjunto de dados, duas novas linhas são adicionadas, sequencialmente, no respectivo arquivo fasta
                        2.4.2.2.1- A primeira linha contém o id da sequência
                        2.4.2.2.2- A segunda linha contém a própria sequência

# A execução dos modelos CNN em níveis isolados
- Foi criado um script para executar testes em um grupo de modelos CNN
    - Esse script permite executar testes combinando um conjunto de hiperparâmetros
        - Esses hiperparâmetros são dos tipos:
            1- level
                -- Nível da classificação taxonômica
            2- batch_sizes
                -- Tamanho dos batches para o treinamento do modelo
                -- Quando selecionada a opção "dynamic", o tamanho dos batches vai variar de acordo com a época e um conjunto de tamanhos pré-estabelecidos na função de treinamento
            3- epochs
                -- Número de épocas do treinamento
            4- models_list
                -- Modelos a serem testados
            5- loss_functions
                -- Funções de custo
            6- Learning_rates
                -- Taxas de aprendizagem 
            7- Optimizers
                -- Otimizadores
    - O script:
        1- Para cada nível taxonômico selecionado:
            1.1- São carregados os datasets de treino e teste gerados anteriormente
            1.2- É criado um objeto SequenceDataset, seguindo a estrutura do objeto Dataset do PyTorch
                1.2.1- As classes usadas para a classificação são selecionadas pela filtragem de valores únicos da união entre as classes de treino e as classes de teste
                    - Idealmente, todas as classes presentes no conjunto de testes deveria estar no conjunto de treino. Mas como a divisão dos conjuntos é feita de forma aleatória, não há como garantir essa condição
                1.2.2- Para garantir consistência nos índices das classe, elas são ordenadas
                1.2.3- É feito o OneHotEncode das classificações das sequências de treino
                1.2.4- É realizado o encode das sequências de treino
                    -- O encode das sequências é feito por um mapeamento de cada base para uma representação vetorial, gerando um vetor de vetores, com dimensão NúmeroDeBases x TamanhoDaSequência (4x900)
                1.2.5- É criado um subdataset para o conjunto de teste
                    1.2.5.1- É feito o OneHotEncode das classificações das sequências de teste
                    1.2.5.2- É feito o encode das sequências de teste
            1.3- Para cada combinação de hiperparâmetros, é gerado um modelo
                1.3.1- O modelo gerado é treinado e testado:
                    1.3.1.1- Em cada época, são calculadas a acurácia e o loss médio das etapas de treino e teste
                    1.3.1.2- Os resultados são adicionados a um csv com as informações da execução de cada época
                    1.3.1.3- Se a época atual tiver melhores resultados que as anteriores e uma acurácia superior a 50%, o modelo pré-treinado é exportado para utilização futura
                1.3.2- As informações do modelo e dos melhores resultados obtidos são salvos em um arquivo referente aos registros dos testes

# Execução do plug-in Feature-Classifier
- Arquivo: ./feature_classifier/qiime_script.sh
- Comandos:
    1- Treinamento:

        qiime tools import --type 'FeatureData[Sequence]' --input-path ../../../../new_data/StratifiedSplit/Isolated/species/pr2_train.fasta --output-path ref-seqs.qza

        qiime tools import --type 'FeatureData[Taxonomy]' --input-format HeaderlessTSVTaxonomyFormat --input-path ../../../../new_data/StratifiedSplit/Isolated/species/pr2_train_taxonomy.txt --output-path ref-taxonomy.qza

        qiime feature-classifier fit-classifier-naive-bayes --i-reference-reads ref-seqs.qza --i-reference-taxonomy ref-taxonomy.qza --o-classifier classifier.qza

    2- Teste:

        qiime tools import --type 'FeatureData[Sequence]' --input-path ../../../../new_data/StratifiedSplit/Isolated/species/pr2_test.fasta --output-path test-seqs.qza

        qiime feature-classifier classify-sklearn --i-classifier classifier.qza --i-reads test-seqs.qza --o-classification test-taxonomy.qza

        qiime metadata tabulate --m-input-file test-taxonomy.qza --o-visualization test-taxonomy.qzv

        qiime tools extract --input-path test-taxonomy.qza --output-path results
