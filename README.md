# Rubix ML

Quelques problèmes classiques de machine learning résolus avec le framework Rubix ML, pour la science !

Quelques liens utiles :

- [Documentation](https://rubixml.github.io/ML/latest/)
- [Github](https://github.com/RubixML/ML)

## Prérequis

Ce projet utilise Docker engine, Docker compose et make.

## Installation du projet

`$ make dev`

`$ make run-example`

Si tout va bien, un petit log devrait afficher :

```bash
array(1) {
  [0]=>
  string(1) "A"
}
```

## Données

Les données d'entraînement et de validation des différents modèles de ce projet ne sont pas versionnées. Il vous faudra
les récupérer en amont, puis les sauvegarder dans le bon répertoire.

### Iris de Fisher ou d'Anderson

Ce projet utilise un fichier CSV.

- Emplacement : `/data/iris/iris.data`
- Séparateur : `,`
- Exemple d'observation : `5.1,3.5,1.4,0.2,Iris-setosa`

Il peut être téléchargé sur le site de l'[UCI](https://archive.ics.uci.edu/dataset/53/iris).

### Base de données MNIST

Ce projet utilise des images bitmap.

- Emplacement: `/data/mnist/trainingSet/[0-9]/*.jpg`
- Format: JPG

Il peut être téléchargé sur [Kaggle](https://www.kaggle.com/datasets/scolianni/mnistasjpg).

## Quelques modèles à tester

### Arbre de décision CART sur les iris de Fisher

`$ make run-iris-cart`

Score attendu en validation : 0.93

### Forêt aléatoire sur les iris de Fisher

`$ make run-iris-random-forest`

Score attendu en validation : 1

### Arbre de décision sur MNIST, pour rire...

`$ make run-mnist-cart`

Score attendu en validation : 0.28

### Perceptron multi couche a deux couches cachées sur MNIST

`$ run-mnist-nn`

Score attendu en validation : 0.97

L'entraînement étant particulièrement long (environ 45 minutes), ce modèle sera persisté dans `/model/mnist-nn-256-128.rbx`

`$ run-mnist-predict`

Infère le modèle persisté à l'étape précédente et effectue une prédiction sur le fichier `/data/test_3.jpg` (non versionné, à vous de jouer). Si tout va bien, le modèle devrait prédire chiffre correspondant au dessin.
