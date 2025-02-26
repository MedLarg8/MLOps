# Variables
VENV_NAME = venv
REQUIREMENTS = requirements.txt
PYTHON = python
PIP = pip
LINTER = flake8
FORMATTER = black
SECURITY = bandit
APP = app.py

# Etape 1: Création de l'environnement virtuel et installation des dépendances
install: $(VENV_NAME)/bin/activate
	$(PIP) install -r $(REQUIREMENTS)


mlflow-start:
	mlflow ui --host 0.0.0.0 --port 5000 &

test-api: 
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

$(VENV_NAME)/bin/activate:
	$(PYTHON) -m venv $(VENV_NAME)
	$(PIP) install --upgrade pip
	touch $(VENV_NAME)/bin/activate

# Etape 2: Vérification du code (Formatage, Qualité, Sécurité)
format:
	$(FORMATTER) .  # Formatte tous les fichiers Python

lint:
	$(LINTER) .  # Vérifie la qualité du code Python

security:
	$(SECURITY) .  # Vérifie les problèmes de sécurité dans le code

# Etape 3: Préparer les données
prepare:
	$(PYTHON) main.py --prepare --file_path merged_churn.csv

# Etape 4: Entraîner le modèle
train:
	$(PYTHON) main.py --train --file_path merged_churn.csv

# Etape 5: Tester le modèle
evaluate:
	$(PYTHON) main.py --evaluate --file_path merged_churn.csv

# Etape 6: Exécuter l'intégralité du pipeline
all: install format lint security prepare train evaluate
