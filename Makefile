.PHONY: build
build:
	docker compose build

.PHONY: install
install:
	docker compose run --rm rubixml composer install

.PHONY: dev
dev: build install

.PHONY: run-example
run-example:
	docker compose run --rm rubixml php src/example.php

.PHONY: run-iris-cart
run-iris-cart:
	docker compose run --rm rubixml php src/iris-cart.php

.PHONY: run-iris-random-forest
run-iris-random-forest:
	docker compose run --rm rubixml php src/iris-random-forest.php

.PHONY: run-mnist-cart
run-mnist-cart:
	docker compose run --rm rubixml php src/mnist-cart.php

.PHONY: run-mnist-nn
run-mnist-nn:
	docker compose run --rm rubixml php src/mnist-nn.php

.PHONY: run-mnist-predict
run-mnist-nn-predict:
	docker compose run --rm rubixml php src/mnist-nn-predict.php

.PHONY: shell
shell:
	docker compose run --rm rubixml bash
