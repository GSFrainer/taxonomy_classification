## ./
    - data_generator.ipynb : Notebook to generate train and test datasets and export the respective files (csv, taxonomy.txt, and fasta)
    - notes.md : Notes about the experiments

## ./CNN
    CNN models and tests
    - ./results
        - ./epochs : Log of the tests epochs
        - ./summarized : Results of the test executions
    - batch_run.ipynb : Notebook for CNN models train and tests

## ./data
    Sequences Dataset(s)
    - cleaned_sequences.csv : Pre-filtered PR2 sequences dataset

## ./feature_classifier
    QIIME2 feature classifier plugin data and tests
    - analysis.ipynb : Notebook to check the accuracy of feature-classifier tests
    - qiime_script.sh : Script to execute train and test with the feature-classifier

## Current Results:

| id | level | batch_size | epochs | model | learning_rate | best_epoch | train_acc_best_epoch | test_acc_best_epoch |
|----|-------|------------|--------|-------|--------------|------------|----------------------|---------------------|
| 8 | class | dynamic | 150 | SimplestCNNClassifier_2layers_Residual | 0.001 | 141.0 | 0.997202 | 0.950846 |
| 9 | class | dynamic | 150 | SimplestCNNClassifier_2layers_Residual | 0.001 | 148.0 | 0.998659 | 0.950279 |
| 3 | class | dynamic | 150 | SimplestCNNClassifier_2layers_Residual | 0.001 | 134.0 | 0.998910 | 0.949632 |
| 2 | class | dynamic | 150 | SimplestCNNClassifier_2layers_Residual | 0.001 | 129.0 | 0.997113 | 0.946473 |
| 6 | class | dynamic | 150 | SimplestCNNClassifier_2layers | 0.001 | 119.0 | 0.999199 | 0.945421 |
| 0 | class | dynamic | 150 | SimplestCNNClassifier_2layers | 0.001 | 109.0 | 0.999307 | 0.944935 |
| 7 | class | dynamic | 150 | SimplestCNNClassifier_2layers | 0.001 | 99.0 | 0.999100 | 0.944773 |
| 1 | class | dynamic | 150 | SimplestCNNClassifier_2layers | 0.001 | 101.0 | 0.999468 | 0.943882 |
| 0 | class | dynamic | 300 | SimplestCNNClassifier0 | 0.001 | 210.0 | 0.999962 | 0.943153 |
| 10 | class | dynamic | 150 | SimplestCNNClassifier_2layers_concat | 0.001 | 104.0 | 0.999118 | 0.942424 |
| 4 | class | dynamic | 150 | SimplestCNNClassifier_2layers_concat | 0.001 | 100.0 | 0.999326 | 0.942182 |
| 11 | class | dynamic | 150 | SimplestCNNClassifier_2layers_concat | 0.001 | 96.0 | 0.999136 | 0.941777 |
| 4 | class | dynamic | 300 | SimplestCNNClassifier5 | 0.001 | 291.0 | 0.999929 | 0.940643 |
| 5 | class | dynamic | 150 | SimplestCNNClassifier_2layers_concat | 0.001 | 87.0 | 0.998570 | 0.940562 |
| 2 | class | dynamic | 300 | SimplestCNNClassifier1 | 0.001 | 198.0 | 0.999942 | 0.939752 |
| 3 | class | dynamic | 300 | SimplestCNNClassifier1 | 0.001 | 189.0 | 0.999981 | 0.935946 |
| 1 | class | dynamic | 300 | SimplestCNNClassifier0 | 0.001 | 112.0 | 0.999788 | 0.929306 |
| 4 | class | dynamic | 300 | SimplestCNNClassifier0_1layerPooling | 0.001 | 258.0 | 0.999910 | 0.909142 |
| 12 | class | dynamic | 300 | SimplestCNNClassifier0_1layerPooling | 0.001 | 140.0 | 0.999847 | 0.908090 |
| 7 | class | dynamic | 300 | SimplestCNNClassifier0_1layer64cPooling | 0.001 | 146.0 | 0.999942 | 0.907604 |
| 1 | class | 10000 | 200 | SimplestCNNClassifier0_1layerPooling | 0.001 | 172.0 | 0.999820 | 0.902583 |
| 5 | class | dynamic | 300 | SimplestCNNClassifier5 | 0.001 | 248.0 | 0.999929 | 0.899911 |
| 4 | class | 10000 | 200 | SimplestCNNClassifier0_1layer64cPooling | 0.001 | 175.0 | 0.999730 | 0.894809 |
| 7 | class | 10000 | 200 | SimplestCNNClassifier5_1layerPooling | 0.001 | 195.0 | 0.999838 | 0.887440 |
| 1 | class | dynamic | 300 | SimplestCNNClassifier0_1layerk4 | 0.001 | 240.0 | 0.999910 | 0.875941 |
| 5 | class | dynamic | 300 | SimplestCNNClassifier0_1layerGELU | 0.001 | 279.0 | 0.999910 | 0.871407 |
| 11 | class | dynamic | 300 | SimplestCNNClassifier0_1layer | 0.001 | 276.0 | 0.999892 | 0.869625 |
| 3 | class | dynamic | 300 | SimplestCNNClassifier0_1layer | 0.001 | 284.0 | 0.999904 | 0.868410 |
| 0 | class | dynamic | 300 | SimplestCNNClassifier0_1layer16 | 0.001 | 237.0 | 0.999897 | 0.867844 |
| 6 | class | dynamic | 300 | SimplestCNNClassifier0_1layer64c | 0.001 | 298.0 | 0.999929 | 0.867196 |
| 8 | class | dynamic | 300 | SimplestCNNClassifier0_1layer16 | 0.001 | 262.0 | 0.999874 | 0.867034 |
| 9 | class | dynamic | 300 | SimplestCNNClassifier0_1layerk4 | 0.001 | 263.0 | 0.999874 | 0.865495 |
| 10 | class | dynamic | 300 | SimplestCNNClassifier0_1layerk2 | 0.001 | 274.0 | 0.999892 | 0.862661 |
| 2 | class | dynamic | 300 | SimplestCNNClassifier0_1layerk2 | 0.001 | 263.0 | 0.999904 | 0.857964 |
| 5 | class | 10000 | 200 | SimplestCNNClassifier5_1layer | 0.001 | 114.0 | 0.999892 | 0.856345 |
| 0 | class | 10000 | 200 | SimplestCNNClassifier0_1layer | 0.001 | 153.0 | 0.999811 | 0.846708 |
| 3 | class | 10000 | 200 | SimplestCNNClassifier0_1layer64c | 0.001 | 169.0 | 0.999748 | 0.846627 |