.PHONY: build
build:
	docker compose build

.PHONY: install
install:
	docker compose run --rm rubixml composer install

.PHONY: run-example
run-example:
	docker compose run --rm rubixml php src/example.php

.PHONY: shell
shell:
	docker compose run --rm rubixml bash