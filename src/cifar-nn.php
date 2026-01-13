<?php

require __DIR__ . '/../vendor/autoload.php';

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Transformers\ImageVectorizer;

$samples = $labels = [];

foreach (['cat', 'dog'] as $class) {
  foreach (glob(__DIR__ . '/../data/CIFAR-10-images/train/' . $class . '/*.jpg') as $file) {
      $samples[] = [imagecreatefromjpeg($file)];
      $labels[] = $class;
  }
}

$dataset = Labeled::build($samples, $labels)
    ->apply(new ImageVectorizer());

var_dump($dataset->sample(4999));
var_dump($dataset->label(4999));
