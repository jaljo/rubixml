.PHONY: build
build:
	docker compose build

.PHONY: install
install:
	docker compose run --rm rubixml composer install

.PHONY: run-example
run-example:
	docker compose run --rm rubixml php src/example.php

.PHONY: run-iris-cart
run-iris-cart:
	docker compose run --rm rubixml php src/iris-cart.php

.PHONY: run-iris-random-forest
run-iris-random-forest:
	docker compose run --rm rubixml php src/iris-random-forest.php

.PHONY: shell
shell:
	docker compose run --rm rubixml bash
