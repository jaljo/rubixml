FROM php:8.2-cli

RUN apt-get update && apt-get install -y \
    git \
    unzip \
    libzip-dev \
		libjpeg62-turbo-dev \
		libpng-dev \
    libicu-dev \
    && docker-php-ext-install zip \
    && docker-php-ext-configure gd --with-jpeg \
    && docker-php-ext-install gd \
    && docker-php-ext-configure intl \
    && docker-php-ext-install intl \
    && rm -rf /var/lib/apt/lists/*

COPY ./docker/php/conf/php.ini /usr/local/etc/php/conf.d/zz-docker.ini

COPY --from=composer:2 /usr/bin/composer /usr/bin/composer

WORKDIR /app
