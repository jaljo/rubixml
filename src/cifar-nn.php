<?php

require __DIR__ . '/../vendor/autoload.php';

use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Loggers\Screen;
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\Transformers\L1Normalizer;
// use Rubix\ML\Transformers\ImageVectorizer;
use Rubix\ML\Transformers\NumericStringConverter;

srand(42);

// $samples = [];
// $labels = [];
// $LIM_SAMPLES = 100;

// foreach (['square', 'circle'] as $class) {
//   foreach (glob(__DIR__ . '/../data/2D_Geometric_Shapes_Dataset/' . $class . '/*.png') as $idx => $file) {
//       if ($idx === $LIM_SAMPLES) {
//         break;
//       }

//       $samples[] = [imagecreatefrompng($file)];
//       $labels[] = $class;
//   }
// }

// $dataset = Labeled::build($samples, $labels)->apply(new ImageVectorizer())->randomize();
// [ $train, $val ] = $dataset->stratifiedSplit(0.8);

$iterator = new CSV(__DIR__ . '/../data/iris/iris.data', header: false);
$dataset = Labeled::fromIterator($iterator)
    ->apply(new NumericStringConverter())
    ->apply(new L1Normalizer())
    ->dropFeature(0)
    ->dropFeature(0)
    ->randomize();

[ $train, $val ] = $dataset->stratifiedSplit(0.8);

$estimator = new MultilayerPerceptron(
  hiddenLayers: [
    new Dense(100),
  ],
);
$estimator->setLogger(new Screen());
$estimator->train($train);

$predictions = $estimator->predict($val);

$accuracy = (new Accuracy())->score($predictions, $val->labels());
var_dump($accuracy);
