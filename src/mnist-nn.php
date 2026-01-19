<?php

require __DIR__ . '/../vendor/autoload.php';

use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Loggers\Screen;
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Transformers\ImageVectorizer;
use Rubix\ML\Transformers\MinMaxNormalizer;

srand(0);

$samples = [];
$labels = [];
$classMap = [ '0' => 'zero', '1' => 'one', '2' => 'two', '3' => 'three' ];

foreach (['0', '1', '2', '3'] as $class) {
  foreach (glob(__DIR__ . '/../data/mnist/trainingSet/trainingSet/' . $class . '/*.jpg') as $idx => $file) {
    $samples[] = [imagecreatefromjpeg($file)];
    $labels[] = $classMap[$class];
  }
}

$dataset = Labeled::build($samples, $labels)
  ->apply(new ImageVectorizer(grayscale: true))
  ->apply(new MinMaxNormalizer())
  ->randomize();

[ $train, $val ] = $dataset->stratifiedSplit(0.8);

$model = new MultilayerPerceptron(
  hiddenLayers: [
    new Dense(256),
    new Activation(new ReLU()),
    new Dense(128),
    new Activation(new ReLU()),
  ],
);
$persister = new Filesystem(__DIR__ . '/../model/mnist-nn.rbx');

$estimator = new PersistentModel(base: $model, persister: $persister);
$estimator->setLogger(new Screen());
$estimator->train($train);
$estimator->save();

$predictions = $estimator->predict($val);

$accuracy = (new Accuracy())->score($predictions, $val->labels());
var_dump($accuracy);
