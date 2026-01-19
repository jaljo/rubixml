<?php

require __DIR__ . '/../vendor/autoload.php';

use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Transformers\ImageVectorizer;
use Rubix\ML\Transformers\MinMaxNormalizer;

srand(0);

$samples = [];
$labels = [];

$classMap = [
  '0' => 'zero',
  '1' => 'one',
  '2' => 'two',
  '3' => 'three',
  '4' => 'four',
  '5' => 'five',
  '6' => 'six',
  '7' => 'seven',
  '8' => 'eight',
  '9' => 'nine',
];

foreach (['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] as $class) {
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

$estimator = new ClassificationTree(maxHeight: 4);
$estimator->train($train);

$predictions = $estimator->predict($val);

$accuracy = (new Accuracy())->score($predictions, $val->labels());
var_dump($accuracy);
