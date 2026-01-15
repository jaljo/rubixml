<?php

require __DIR__ . '/../vendor/autoload.php';

use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Loggers\Screen;
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\Transformers\ImageVectorizer;
use Rubix\ML\Transformers\MinMaxNormalizer;

srand(0);

$samples = [];
$labels = [];
$LIM_SAMPLES = 2000;

$classMap = [
  '3' => 'three',
  '9' => 'nine',
  '0' => 'zero',
  '8' => 'eight',
  '5' => 'five',
];

foreach (['3', '5', '8', '9'] as $class) {
  foreach (glob(__DIR__ . '/../data/mnist/trainingSet/trainingSet/' . $class . '/*.jpg') as $idx => $file) {
      if ($idx === $LIM_SAMPLES) {
        break;
      }

      $samples[] = [imagecreatefromjpeg($file)];
      $labels[] = $classMap[$class];
  }
}

$dataset = Labeled::build($samples, $labels)
  ->apply(new ImageVectorizer())
  ->apply(new MinMaxNormalizer())
  ->randomize();
[ $train, $val ] = $dataset->stratifiedSplit(0.8);

$estimator = new MultilayerPerceptron(
  hiddenLayers: [
    new Dense(256),
    new Activation(new Sigmoid()),
    new Dense(128),
    new Activation(new Sigmoid()),
    new Dense(64),
    new Activation(new Sigmoid()),
    new Dense(32),
  ],
);
$estimator->setLogger(new Screen());
$estimator->train($train);

$predictions = $estimator->predict($val);

$accuracy = (new Accuracy())->score($predictions, $val->labels());
var_dump($accuracy);
