# LLMOPS - Language Model Operations
## Project Overview
LLMOPS is a pipeline designed to handle various stages of large language model (LLM) operations. This includes dataset preparation, inference, evaluation, fine-tuning, and synthetic data generation. The architecture is built to streamline these processes, allowing for more efficient management and deployment of LLMs.
### Key Features:
* `Fine-Tuning`: Adjust LLMs for specific tasks using custom datasets.
* `Dataset Merging`: Combine datasets into a single, unified training or evaluation dataset.
* `Synthetic Data Generation`: Create synthetic data to enhance model training.
* `Batch Inference`: Perform inference on large datasets efficiently.
* `Evaluation`: Measure model performance on predefined criteria.
  
The architecture of this project is represented in the following images:

![LLMOPS Architecture](Data/LLMOPs_workflow.jpeg) ![LLMOPS Architecture](Data/LLMOPS_pipeline.jpeg)

## Installation Requirements
To set up and run this project, the following Python packages are required:
```
python3.12 -m pip install openai==0.28
 python3.12 -m pip install git+https://github.com/huggingface/alignment-handbook.git@main
pip install -U transformers
pip install -U peft
pip install -U accelerate
pip install -U bitsandbytes
pip install -U datasets
pip install time
pip install tqdm
pip install pyyaml
```
Additionally, log into Hugging Face to access open-source models:
``` !huggingface-cli login ```

## Commands to Run Various Stages
### Batch Inference
Perform batch inference over a large dataset using the following command:
```
python batch_inference.py --from-config config/batch_inference.yaml

```
Description: The `batch_inference.py` script processes input data in batches, leveraging the model specified in `batch_inference.yaml`. This allows for efficient handling of large datasets.
### Model Evaluation
To evaluate the performance of the models on specific criteria, use this command:
```
python evaluation.py --from-config config/evaluation.yaml
```
Description: The `evaluation.py` script runs multiple evaluations of the model based on predefined metrics in the `evaluation.yaml` file. The results are then used to optimize the model or compare different models.
### Synthetic Data Generation
Generate synthetic data to augment the training dataset using this command:
``` 
python data_gen.py --from-config config/synthdatagen.yaml
```
Description: The `data_gen.py` script creates synthetic data from prompts specified in the `synthdatagen.yaml` file, which can be used to diversify the model’s training data or test its robustness.

### Fine-Tuning with Hugging Face

To fine-tune models using the Hugging Face `alignment-handbook`, clone the repository and copy your configuration:
```
# Clone the repository
git clone https://github.com/huggingface/alignment-handbook.git

# Create custom config directory and copy the configuration file of the model you want to fine-tune
mkdir -p alignment-handbook/recipes/custom/
cp "/home/devadmin/LLMOPS/config/sample_config.yaml" alignment-handbook/recipes/custom/config.yaml

# Navigate to the repository directory
cd alignment-handbook

# Install the required packages
python -m pip install .

# Install Accelerate and other dependencies
pip install accelerate
```
To launch the training using `accelerate`, run the following command:
```
accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/custom/config.yaml

```
### Monitoring and Logging
All model training, evaluation, and inference operations are monitored and logged using ``Langfuse``. Langfuse provides real-time insights into the performance of your models.

## Choix Technologiques

Le choix des technologies pour le développement du framework LLMOPS a été soigneusement réfléchi afin d'assurer un équilibre optimal entre performance, flexibilité, évolutivité et compatibilité avec les pratiques et outils de pointe dans le domaine de l'intelligence artificielle.

### Langage de Programmation : Python

Python a été choisi comme langage principal pour l'implémentation du framework. Il est largement adopté dans le domaine de l'IA et offre un écosystème riche de bibliothèques pour la manipulation des données, l'entraînement des modèles, et l'intégration avec des services cloud et API. Python permet également un prototypage rapide et une grande flexibilité grâce à sa syntaxe simple et sa vaste communauté de développeurs.

### Frameworks et Outils Utilisés

- **Hugging Face Transformers** : Ce framework est étroitement intégré dans LLMOPS pour gérer les modèles de langage open source. Hugging Face est reconnu pour sa collection de modèles pré-entraînés et ses outils puissants pour le fine-tuning, l'inférence et l'évaluation des LLMs. Le framework facilite également l'intégration des datasets et des modèles via des APIs.

- **Hugging Face's Alignment Handbook** : Pour améliorer et rationaliser le processus de fine-tuning des LLMs, LLMOPS utilise l'Alignment Handbook de Hugging Face. Cet outil fournit des lignes directrices et des méthodologies éprouvées pour aligner les modèles sur des objectifs spécifiques.

- **DeepInfra** : Pour répondre aux besoins en modèles open source avec des APIs facilement accessibles, LLMOPS utilise les services de DeepInfra. Ce fournisseur offre des modèles open source déployés via des APIs, permettant une intégration rapide et une scalabilité adaptée aux besoins des entreprises.

- **OpenAI GPT Models** : LLMOPS intègre également les modèles GPT d'OpenAI via des APIs, tels que GPT-3.5 et GPT-4, offrant des capacités avancées de génération de texte, de compréhension contextuelle et de résolution de problèmes complexes.

- **Langfuse** : Pour le monitoring et le logging des interactions avec les LLMs, LLMOPS s'appuie sur Langfuse, qui permet un suivi en temps réel des performances.

- **Elasticsearch** : Utilisé comme solution de stockage et de recherche pour gérer efficacement les grands volumes de données générées et traitées par LLMOPS.

### Gestion des Configurations et CLI

- **Fichiers de Configuration YAML/TOML** : La gestion des configurations est réalisée via des fichiers YAML et TOML, qui permettent une structuration claire des paramètres nécessaires au fonctionnement du framework.

- **Interface en Ligne de Commande (CLI) et Argparse** : LLMOPS est conçu pour être opéré principalement via une CLI, utilisant le module `argparse` pour gérer les arguments de ligne de commande.

## Fonctionnalités du LLMOPS

### Couche de Configuration dans le Framework LLMOPS

La couche de configuration dans le framework LLMOPS est conçue pour être hautement configurable par l'utilisateur, permettant une grande flexibilité et adaptabilité à travers différents cas d'utilisation. En modifiant simplement les fichiers YAML et TOML, les utilisateurs peuvent contrôler l'ensemble du cycle de vie des LLMs, de l'inférence à l'évaluation et à la génération de données synthétiques. Cette couche abstrait la complexité des modifications de code directes, la rendant accessible même aux utilisateurs ayant une expérience de codage minimale tout en conservant les capacités puissantes nécessaires pour une gestion robuste des modèles d'IA.

#### Exemples de fichiers de configuration

1. **batch_inference.yaml**
   - **Description** : Ce fichier de configuration gère le processus d'inférence par lots, responsable de l'exécution des LLMs sur de grands ensembles de données.
   - **Paramètres Clés** :
     - `ft_model_id` : Identifiant du modèle affiné.
     - `ft_model_revision` : La révision du modèle à utiliser, souvent spécifiée comme une branche ou un commit dans les dépôts versionnés.
     - `ft_model_config_path` : Chemin vers le fichier de configuration du modèle, qui peut définir les paramètres du tokenizer, les hyperparamètres du modèle, etc.
     - `load_in_8bit`, `load_in_4bit` : Indicateurs booléens pour spécifier si le modèle doit être chargé en mode de précision réduite pour économiser de la mémoire.
     - `test_ds_id` : Identifiant pour l'ensemble de données de test utilisé pour les inférences.
     - `batch_infer_data_preprocess_bs` : Taille des lots pour le prétraitement des données lors de l'inférence.
     - `inference_bs` : Taille des lots pour l'exécution des inférences.
     - `lm_response_ds_id` : Identifiant de l'emplacement où les réponses du LLM après l'inférence seront stockées.
     - `lm_response_append` : Indicateur booléen pour déterminer si les réponses du LLM doivent être ajoutées à l'ensemble de données existant.

2. **evaluation.yaml**
   - **Description** : Ce fichier configure le processus d'évaluation, où les performances des LLMs sont évaluées à l'aide de diverses métriques.
   - **Paramètres Clés** :
     - `service_model_name` : Nom du modèle de service évalué, comme "gpt-4" ou "gpt-3.5-turbo".
     - `rate_limit_per_minute` : Limite de débit pour les appels API, garantissant que le modèle ne dépasse pas les quotas d'utilisation.
     - `prompt_tmpl_path` : Chemin vers le fichier TOML contenant les modèles de prompt pour l'évaluation.
     - `lm_response_ds_id` : Identifiant de l'ensemble de données où les réponses du LLM pour l'évaluation sont stockées.
     - `eval_data_preprocess_bs` : Taille des lots pour le prétraitement des données dans le pipeline d'évaluation.
     - `eval_repeat` : Nombre de répétitions du processus d'évaluation, utile pour obtenir des résultats moyens.
     - `eval_workers` : Nombre de travailleurs parallèles à utiliser lors de l'évaluation.
     - `langfuse_public_key`, `langfuse_secret_key`, `langfuse_host` : Identifiants et informations sur l'hôte pour l'intégration avec Langfuse pour le monitoring et le logging.
     - `session_id` : Identifiant unique de session pour le suivi des sessions d'évaluation dans Langfuse.

3. **synthdatagen.yaml**
   - **Description** : Ce fichier de configuration est utilisé pour générer des données synthétiques, qui peuvent être utilisées pour l'affinement ou les tests de modèles.
   - **Paramètres Clés** :
     - `service_model_name` : Spécifie le modèle à utiliser pour générer des données synthétiques, comme "gpt-3.5-turbo".
     - `rate_limit_per_minute` : Limite de débit pour les appels API.
     - `prompt_tmpl_path` : Chemin vers le fichier TOML du modèle de prompt utilisé pour la génération de données synthétiques.
     - `reference_ds_id` : Identifiant de l'ensemble de données de référence servant de guide pour générer des données synthétiques.
     - `num_samples` : Nombre d'échantillons de données synthétiques à générer.
     - `gen_workers` : Nombre de travailleurs pour paralléliser le processus de génération de données.
     - `synth_ds_id` : Identifiant de l'ensemble de données où les données synthétiques seront stockées.
     - `synth_ds_append` : Indicateur booléen indiquant si les nouvelles données synthétiques doivent être ajoutées à un ensemble de données existant.

4. **prompts.toml**
   - **Description** : Ce fichier TOML contient les modèles de prompts utilisés à différents stades du cycle de vie des LLMs, tels que l'évaluation et la génération de données synthétiques.
   - **Sections Clés** :
     - `eval` : Définit la structure et le contenu des prompts utilisés lors de l'évaluation des modèles. Cela inclut les instructions et les attentes pour les réponses du modèle, ainsi que la structure de notation pour la pertinence, la complétude, la grammaire, et plus encore.
     - `synth_data_gen` : Fournit un modèle pour générer des paires instruction-réponse synthétiques, adaptées à des contextes spécifiques comme l'éducation.


### Couche de Traitement des Données

La couche de traitement des données dans LLMOPS est conçue pour être flexible et puissante, permettant aux utilisateurs de générer des données synthétiques et de fusionner des datasets avec facilité. Cette couche s'intègre étroitement avec l'écosystème Hugging Face.

#### Génération de Données Synthétiques (`data_gen.py`)

- **Description** : Ce script est central pour la génération de données synthétiques, qui peuvent être utilisées pour l'affinement des modèles, les tests ou la création de données supplémentaires pour l'entraînement.

- **Fonctions Principales** :
  - `synth_data_generation` : Génère des datasets synthétiques qui imitent les caractéristiques de l'ensemble de données original.

- **Entrées Nécessaires** :
  - `reference_ds_id` : L'ensemble de données à partir duquel les données synthétiques sont dérivées.
  - `num_samples` : Spécifie combien d'échantillons de données synthétiques doivent être générés.
  - `topic` : Le sujet thématique des données générées.
  - `prompt_tmpl_path` : Chemin vers le fichier TOML contenant les modèles de prompts.

#### Fusion de Datasets (`dataset_merge.py`)

- **Description** : Ce script se concentre sur la fusion de plusieurs datasets en un seul ensemble cohérent.

- **Fonctions Principales** :
  - `merge_datasets` : La fonction principale responsable de la fusion de deux ensembles de données en un seul.

- **Entrées Nécessaires** :
  - `first_ds_id` et `second_ds_id` : Identifiants pour les ensembles de données à fusionner.
  - `result_ds_id` : Identifiant pour l'ensemble de données fusionné résultant.

### Couche d'Inférence

La couche d'inférence dans le framework LLMOPS est conçue pour être à la fois flexible et efficace, prenant en charge une large gamme de LLMs et de cas d'utilisation.

#### Inférence par Lots (`batch_inference.py`)

- **Description** : Ce script est l'épine dorsale du processus d'inférence, responsable de l'exécution d'inférences par lots sur les données d'entrée à l'aide des LLMs spécifiés.

- **Fonctions Principales** :
  - `batch_inference` : Exécute les inférences par lots.

- **Entrées Nécessaires** :
  - `ft_model_id` : Identifiant du modèle affiné à utiliser.
  - `ft_model_revision` : Indique la version du modèle à utiliser.
  - `test_ds_id` : Identifiant pour l'ensemble de données utilisé pour les inférences.
  - `batch_infer_data_preprocess_bs` : Taille des lots pour le prétraitement des données lors de l'inférence.
  - `inference_bs` : Taille des lots pour l'exécution des inférences.

#### Fonctionnalité d'Inférence dans LLMOps

#### 1. Personnalisation du Modèle
Configurez votre modèle en fonction de vos besoins spécifiques.

##### Paramètres Clés
- `model_id` : Identifiant unique du modèle.
- `model_revision` : Révision spécifique du modèle.
- `model_config_path` : Chemin d'accès au fichier de configuration du modèle.

#### 2. Optimisation de la Performance
Améliorez les performances de chargement et d'inférence du modèle.

##### Paramètres Clés
- `load_in_8bit` : Charge le modèle en utilisant une précision de 8 bits.
- `load_in_4bit` : Charge le modèle en utilisant une précision de 4 bits.

#### 3. Gestion des Données de Test
Organisez et gérez vos jeux de données pour des tests efficaces.

##### Paramètres Clés
- `test_ds_id` : Identifiant du jeu de données de test.
- `test_ds_split` : Fraction du jeu de données de test à utiliser.

#### 4. Prétraitement et Inférence par Lots
Optimisez le prétraitement des données et les inférences en lots.

##### Paramètres Clés
- `batch_infer_data_preprocess_bs` : Taille de batch pour le prétraitement des données d'inférence.
- `inference_bs` : Taille de batch pour l'inférence.

#### 5. Enregistrement et Gestion des Réponses du Modèle
Gérez et enregistrez les réponses générées par le modèle.

##### Paramètres Clés
- `lm_response_ds_id` : Identifiant du jeu de données de réponses du modèle.
- `lm_response_ds_split` : Fraction du jeu de données pour les réponses du modèle.
- `lm-response-append` : Mode d'ajout des réponses.
- `push_lm_responses_to_langfuse` : Envoi des réponses générées vers Langfuse pour suivi.


### Couche d'Évaluation

La couche d'évaluation dans le framework LLMOPS est conçue pour offrir un système robuste et flexible pour évaluer les performances des LLMs.

#### Script d'Évaluation (`evaluation.py`)

- **Description** : Ce script gère le processus global d'évaluation, en interagissant avec le service Langfuse pour surveiller et consigner les résultats de l'évaluation.

- **Fonctions Principales** :
  - `load_config` : Charge les paramètres de configuration à partir d'un fichier YAML.
  - `log_to_langfuse` : Consigne les résultats d'évaluation dans Langfuse.

- **Entrées Nécessaires** : Les utilisateurs doivent fournir les identifiants des ensembles de données et d'autres détails de configuration via un fichier YAML.

#### Fonctionnalité d'Evaluation dans LLMOps

#### 1. Sélection et Utilisation du Modèle de Service
Choisissez et configurez le modèle de service pour les évaluations avec des limitations de taux.

##### Paramètres Clés
- `service_model_name` : Nom du modèle de service utilisé pour l'évaluation.
- `rate_limit_per_minute` : Limite de requêtes par minute pour le modèle de service.

#### 2. Personnalisation des Prompts d'Évaluation
Configurez le modèle de prompts d'évaluation en fonction de vos besoins.

##### Paramètres Clés
- `prompt_tmpl_path` : Chemin d'accès au fichier de template de prompts.

#### 3. Gestion des Données de Réponses et d'Évaluation
Organisez et suivez les jeux de données de réponses et d'évaluation pour une analyse efficace.

##### Paramètres Clés
- `lm_response_ds_id` : Identifiant du jeu de données de réponses du modèle.
- `lm_response_ds_split` : Fraction du jeu de données pour les réponses.
- `eval_ds_id` : Identifiant du jeu de données d'évaluation.
- `eval_ds_split` : Fraction du jeu de données pour l'évaluation.

#### 4. Optimisation des Ressources pour l'Évaluation
Améliorez les performances d'évaluation en optimisant les ressources.

##### Paramètres Clés
- `eval_data_preprocess_bs` : Taille de batch pour le prétraitement des données d'évaluation.
- `eval_repeat` : Nombre de répétitions pour chaque évaluation.
- `eval_workers` : Nombre de travailleurs parallèles pour l'évaluation.

#### 5. Intégration avec Langfuse pour le Monitoring
Connectez-vous à Langfuse pour le suivi des performances d'évaluation.

##### Paramètres Clés
- `langfuse_public_key` : Clé publique pour l'authentification avec Langfuse.
- `langfuse_secret_key` : Clé secrète pour l'authentification avec Langfuse.
- `langfuse_host` : Hôte de Langfuse.
- `session_id` : Identifiant de la session pour le suivi.
- `prompt_langfuse` : Indicateur pour envoyer les prompts à Langfuse.


### Couche de Fine-Tuning

La couche de Fine-Tuning dans le framework LLMOPS est un élément clé qui permet aux entreprises d'ajuster les modèles de langage de grande taille (LLMs) selon leurs besoins spécifiques.

#### Description de la Couche

Le fine-tuning dans LLMOPS est conçu pour adapter les modèles LLMs aux cas d'utilisation particuliers de l'entreprise. Cette couche est construite pour être flexible, permettant une modification aisée des paramètres d'entraînement et une gestion efficace des ressources.

- **Fonctions Principales** :
  - Chargement et configuration du modèle.
  - Prétraitement des données et mélange des datasets.
  - Exécution du fine-tuning via le script `scripts/run_sft.py`.
  - Suivi des performances et logging via des outils comme TensorBoard.

#### Techniques Utilisées

- **Low-Rank Adaptation (LoRA)** : Permet un fine-tuning efficace sans nécessiter la réinitialisation complète du modèle.
- **Support Multi-GPU avec Accelerate** : Optimise l'utilisation des ressources matérielles.

#### Fonctionnalité de Fine-tuning dans LLMOps

#### 1. Fonctionnalité de Fine-tuning dans LLMOps
Ajustez et améliorez les modèles en fonction de jeux de données spécifiques pour des performances accrues.

#### 2. Configuration du Modèle
Configurez le modèle de base pour le fine-tuning.

##### Paramètres Clés
- `model_name_or_path` : Chemin ou nom du modèle à fine-tuner.
- `model_revision` : Révision du modèle à utiliser.
- `torch_dtype` : Type de données PyTorch pour les poids du modèle (ex. : `float32`, `bf16`).

#### 3. Optimisation avec LoRA (Low-Rank Adaptation)
Améliorez l'efficacité du fine-tuning en utilisant LoRA pour des modèles plus légers.

##### Paramètres Clés
- `load_in_4bit` : Charge le modèle en utilisant une précision de 4 bits.
- `use_peft` : Active l'utilisation de PEFT (Parameter-Efficient Fine-Tuning).
- `lora_r` : Paramètre de rang pour LoRA.
- `lora_alpha` : Hyperparamètre alpha pour LoRA.
- `lora_dropout` : Taux de dropout pour LoRA.
- `lora_target_modules` : Modules cibles pour appliquer LoRA.

#### 4. Préparation et Mixage des Données
Configurez et mélangez les jeux de données pour l'entraînement.

##### Paramètres Clés
- `chat_template` : Modèle de chat à utiliser pour le format des données.
- `dataset_mixer` : Méthode pour mélanger différents jeux de données.
- `dataset_splits` : Fractionnement des jeux de données.
- `preprocessing_num_workers` : Nombre de travailleurs pour le prétraitement des données.

#### 5. Configuration de l'Entraîneur (SFT Trainer)
Ajustez les paramètres de l'entraîneur pour le fine-tuning du modèle.

##### Paramètres Clés
- `bf16` : Utilise le format `bfloat16` pour l'entraînement.
- `do_eval` : Effectue une évaluation pendant l'entraînement.
- `evaluation_strategy` : Stratégie d'évaluation (par ex., `steps` ou `epoch`).
- `gradient_accumulation_steps` : Nombre de pas d'accumulation de gradient.
- `gradient_checkpointing` : Active le gradient checkpointing pour économiser la mémoire.
- `learning_rate` : Taux d'apprentissage pour le fine-tuning.

## 6. Gestion des Sorties et du Logging
Configurez la gestion des sorties et le suivi des logs pendant l'entraînement.

### Paramètres Clés
- `hub_model_id` : Identifiant pour publier le modèle sur le hub.
- `output_dir` : Répertoire pour sauvegarder les sorties.
- `logging_strategy` : Stratégie de logging (par ex., `steps` ou `epoch`).
- `report_to` : Destination pour les rapports de suivi (ex., `tensorboard`, `wandb`).
- `save_strategy` : Stratégie de sauvegarde (par ex., `steps` ou `epoch`).



### Couche de Monitoring et Logging

La couche de monitoring et de logging dans le framework LLMOPS est un système robuste conçu pour assurer un suivi continu et un logging des performances des modèles.

#### Intégration du Monitoring Langfuse

- **Fonctions Principales** :
  - `get_eval_prompt_tmpl` : Récupère les modèles de prompt d'évaluation.
  - Gestion des traces et logging des scores.

#### Techniques Utilisées

- **Feedback en temps réel** : Langfuse fournit un feedback en temps réel sur les performances des modèles, permettant au framework d'ajuster dynamiquement les configurations.
- **Logging et Traçabilité Détaillés** : Chaque interaction avec le modèle est loguée avec une grande granularité, permettant une analyse approfondie et un dépannage lorsque nécessaire. Ce niveau détaillé de logging est essentiel pour comprendre comment les modèles performent et comment ils peuvent être améliorés.
- **Configuration Dynamique** : L'utilisation de Langfuse peut être activée ou désactivée en fonction de la configuration, permettant ainsi une flexibilité dans le degré de monitoring et de logging en fonction du cas d'utilisation.

### Couche d'Interface Utilisateur

La couche d'interface utilisateur dans le framework LLMOPS offre un moyen puissant mais accessible pour les utilisateurs d'interagir avec le framework.

#### Interface en Ligne de Commande (CLI)

- **Fonctionnalités Clés** :
  - Permet aux utilisateurs d'interagir facilement avec les scripts d'inférence, d'évaluation, et de logging.
  - Utilise le module `argparse` pour gérer les arguments de ligne de commande.

- **Commandes Utilisateur Clés** :
  - Saisie de la clé API pour authentifier les requêtes.
  - Sélection du modèle à utiliser et gestion des sorties.

#### Facilité d'Utilisation

La CLI abstrait la complexité des processus sous-jacents, rendant le framework accessible même aux utilisateurs non techniques. Cela est réalisé grâce aux fonctionnalités suivantes :

- **Commandes Simplifiées** : La CLI permet aux utilisateurs d'exécuter des tâches complexes, telles que l'exécution d'inférences ou d'évaluations, avec des commandes simples et intuitives.
- **Flexibilité de Configuration** : La CLI est conçue pour fonctionner de manière transparente avec les fichiers de configuration (YAML/TOML) utilisés dans le framework.
- **Messages d'Erreur Conviviaux** : Lorsque les utilisateurs fournissent des arguments incorrects ou conflictuels, la CLI fournit des messages d'erreur clairs et exploitables, aidant les utilisateurs à corriger rapidement leurs entrées et à poursuivre leurs tâches.

