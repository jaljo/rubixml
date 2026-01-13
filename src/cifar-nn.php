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
$LIM_SAMPLES = 200;

foreach (['cat', 'airplane'] as $class) {
  foreach (glob(__DIR__ . '/../data/CIFAR-10-images/train/' . $class . '/*.jpg') as $idx => $file) {
      if ($idx === $LIM_SAMPLES) {
        break;
      }

      $samples[] = [imagecreatefromjpeg($file)];
      $labels[] = $class;
  }
}

$dataset = Labeled::build($samples, $labels)
  ->apply(new ImageVectorizer())
  ->apply(new MinMaxNormalizer())
  ->randomize();
[ $train, $val ] = $dataset->stratifiedSplit(0.8);

$estimator = new MultilayerPerceptron(
  hiddenLayers: [
    new Dense(200),
    // new Activation(new Sigmoid()),
  ],
);
$estimator->setLogger(new Screen());
$estimator->train($train);

$predictions = $estimator->predict($val);

$accuracy = (new Accuracy())->score($predictions, $val->labels());
var_dump($accuracy);
