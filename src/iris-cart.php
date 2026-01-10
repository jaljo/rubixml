<?php

require __DIR__ . '/../vendor/autoload.php';

use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Transformers\NumericStringConverter;

// srand(42);

$iterator = new CSV(__DIR__ . '/../data/iris/iris.data', header: false);
$dataset = Labeled::fromIterator($iterator)
    ->apply(new NumericStringConverter())
    ->dropFeature(0)
    ->dropFeature(0)
    ->randomize()
;

[ $train, $val ] = $dataset->stratifiedSplit(0.8);

$estimator = new ClassificationTree(maxHeight: 4);
$estimator->train($train);

$predictions = $estimator->predict($val);

$accuracy = (new Accuracy())->score($predictions, $val->labels());

var_dump($accuracy);

