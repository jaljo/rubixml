<?php

require __DIR__ . '/../vendor/autoload.php';

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Transformers\ImageVectorizer;

$image = imagecreatefromjpeg(__DIR__ . '/../data/test_3.jpg');
$samples = [[$image]];
$dataset = Unlabeled::build($samples)
  ->apply(new ImageVectorizer(grayscale: true));

$vector = $dataset->sample(0);
$normalizedVector = array_map(fn (int $pixel) => $pixel / 255.0, $vector);
$normalizedDataset = new Unlabeled([$normalizedVector]);

$estimator = PersistentModel::load(new Filesystem(__DIR__ . '/../model/mnist-nn-256-128.rbx'));

$predictions = $estimator->predict($normalizedDataset);

var_dump($predictions);
