# Rapport Data Prep

## Installation

`python -m venv .venv`
`source .venv/bin/activate`
`pip install -r requirements.txt`

## Fichiers

-   Notebook exploratoire : `Pages_Sebastian_1_notebook_exploratoire_022024.ipynb`
-   Notebook prediction CO2 : `Pages_Sebastian_3_notebook_prediction_022024.ipynb`
-   Notebook prediction Energie : `Pages_Sebastian_3_notebook_prediction_022024.ipynb`
-   Notebook presentation : `Pages_Sebastian_4_presentation_022024.pdf`


```sh
my_data_science_project/
│
├── README.md               # Project documentation
├── project_documentation.pdf
├── requirements.txt        # Python dependencies
│
├── data/
│   ├── raw/                # Raw data files
│   ├── processed/          # Processed data files
│   └── results/            # Output results
│
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── data_preprocessing.ipynb
│   └── model_training.ipynb
│
├── scripts/
│   ├── data_processing.py  # Python scripts for data processing
│   ├── model.py            # Python scripts for model definition
│   └── evaluation.py       # Python scripts for evaluation metrics
│
├── cache/                  # Directory for cached data
│   ├── cached_data.pkl     # Example: pickled cached data
│   └── cached_results.npy  # Example: cached results in numpy format
│
├── images/                 # Directory for images used in notebooks or reports
│   ├── plot1.png
│   └── plot2.png
│
├── models/                 # Directory for trained model artifacts
│   ├── model1.pkl          # Example: serialized model
│   └── model2.h5           # Example: model saved in HDF5 format
│
├── src/                    # Source code directory
│   ├── __init__.py         # Makes src a module
│   ├── data_processing.py  # Reusable data processing functions
│   ├── model.py            # Reusable model classes or functions
│   └── evaluation.py       # Reusable evaluation functions
│
├── environment.yml         # Conda environment file
└── .gitignore              # Git ignore file
```


Outils Open Source pour élaborer une plateforme MLOps
● MLFlow pour la gestion “d’expériences” et leur tracking lors de la
phase d’entraînement des modèles, ainsi que la visualisation des
résultats avec MLFlow UI
● MLFlow pour le stockage centralisé des modèles dans un “model
registry” et le serving
● Git, logiciel de version de code, pour suivre les modifications du
code final de l’API de prédiction à déployer
● Github pour stocker et partager sur le cloud le code de l’API,
alimenté par un “push” Git et ainsi assurer une intégration continue
● Github Actions pour le déploiement continu et automatisé du code
de l’API sur le cloud
● Pytest (ou Unittest) pour concevoir les tests unitaires et les
exécuter de manière automatisée lors du build réalisé par Github
Actions