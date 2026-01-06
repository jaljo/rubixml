<?php

require __DIR__ . '/../vendor/autoload.php';

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Classifiers\KNearestNeighbors;

$samples = [
    [1, 2],
    [2, 1],
    [4, 5],
    [5, 4],
];

$labels = [
    'A',
    'A',
    'B',
    'B',
];

$dataset = new Labeled($samples, $labels);

$estimator = new KNearestNeighbors(3);
$estimator->train($dataset);

$toPredict = new Unlabeled([
    [1.5, 1.5]
]);

var_dump($estimator->predict($toPredict));
