# A deep learning model for the classification of living organisms using the 18S rRNA segment of the genetic sequence

A Master's thesis project developing and evaluating a deep learning (CNN) approach for taxonomic classification of genetic sequences, with comparative analysis against QIIME2's feature-classifier plugin.

## 📖 Overview

This repository contains the implementation and some experiments for a deep learning model, CNN based, designed to classify taxonomic information from 18S rRNA genetic sequences. The project compares the performance of the custom deep learning approach against the established QIIME2 feature-classifier plugin using data from the PR2 platform.

## ✍️ Author

- **Name**: Gustavo Savi Frainer
- **LinkedIn**: [gsfrainer](https://www.linkedin.com/in/gsfrainer/)

## ✅ Results Summary

The developed model consistently outperformed the reference tool in terms of accuracy across all classification levels, based on the dataset used. The table below presents the mean accuracies, while a more detailed comparison (including minimum, mean, and maximum values) can be found in the associated thesis document at the end of this page.

* Experiments Mean Acurracies Comparison:

|                   |                   | Mean Accuracy |               |
|-------------------|-------------------|---------------|---------------|
|**Taxonomy Level** | **Sampling**      | **Reference** | **CNN**       |
|Class              | Simple random     | 94.67%        | 99.00%        |
|Class              | Stratified        | 94.77%        | 99.05%        |
|Order              | Simple random     | 94.92%        | 98.52%        |
|Order              | Stratified        | 94.81%        | 98.54%        |
|Family             | Simple random     | 94.19%        | 97.56%        |
|Family             | Stratified        | 94.23%        | 97.58%        |
|Genus              | Simple random     | 82.35%        | 88.80%        |
|Genus              | Stratified        | 82.55%        | 89.52%        |
|Species            | Simple random     | 35.63%        | 90.88%        |
|Species            | Stratified        | 30.37%        | 91.71%        |



## 📦 Dataset

- **Source**: [PR2 (Protist Ribosomal Reference) Database](https://pr2-database.org/)
- **Sequence Type**: 18S rRNA sequences
- **Purpose**: Taxonomic classification training and evaluation

## 🔬 Baseline Comparison
- **Tool**: [QIIME2 feature-classifier plugin](https://docs.qiime2.org/2024.10/tutorials/feature-classifier/)
- **Purpose**: Generate reference classification results for performance comparison


## 📁 Project Structure

### Data (`./data/`)

- `cleaned_sequences.csv`: The complete dataset of genetic sequences used across all experiments.
- `data_generator.ipynb`: Script for generating train and test sets for experiments.


### Reference Experiments (`./feature-classifier/`)

- `qiime_script.sh`: Bash script to execute classification using the QIIME2 feature-classifier plugin.
- `analysis.ipynb`: Reports classification results (correct, incorrect classifications, accuracy).
- `times.ipynb`: Provides time metrics for the reference experiments.


### CNN Tests (`./CNN/`)

- `analysis.ipynb`: Script to generate summaries and insights from CNN results.
- `batch_run.ipynb`: Complete CNN experiment pipeline including model definition, training, and evaluation.
- `times.ipynb`: Script for analyzing execution times.
- `Models/`: Folder with some of the pre-treined models
- `results/`: Folder that contains the logs of the experiments executions


## 📚 Dependencies

* Python >= 3.10
* QIIME2 >= 2024.10.1 (for reference experiments only)
* PyTorch >= 2.4.0
* Pandas, Numpy, Scikit-learn
* Jupyter Notebook


## 🚀 How to Run

### Setup
```bash
## Clone the repository
git clone https://github.com/GSFrainer/taxonomy_classification.git
cd taxonomy_classification

## Install dependencies
pip install -r requirements.txt
```

> **Note:** 
> For QIIME2 setup, follow the official instructions:
> [https://docs.qiime2.org/2024.10/install/](https://docs.qiime2.org/2024.10/install/)

### Generate Train/Test Datasets:
   ```bash
   # Open and run the cells in:
   data/data_generator.ipynb
   ```

### Run Reference Experiments (QIIME2):
   ```bash
   cd feature-classifier
   bash qiime_script.sh
   ```

### Run CNN Experiments:

   ```bash
   # Open and run:
   CNN/batch_run.ipynb
   ```


## 🎓 Thesis Information

- **Title**: Um modelo de deep learning para classificação de organismos vivos utilizando o segmento 18S rRNA da sequência genética
- **Author**: Gustavo Savi Frainer
- **Institution**: Pontifícia Universidade Católica do Rio Grande do Sul (PPGCC/PUCRS)
- **Year**: 2025
- **Advisor**: Dr. Duncan Dubugras Alcoba Ruiz
- **Link**: [PUCRS - Digital Library of Theses and Dissertations](https://tede2.pucrs.br/tede2/handle/tede/11711)


## 💡 Notes

> * The results and methods are part of an completed research project.
> * If you use this work, please cite appropriately or contact the author.


