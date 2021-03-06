
include .env

ifdef http_proxy
BUILD_ARGS += --build-arg="http_proxy=$(http_proxy)"
RUN_ARGS += --env="http_proxy=$(http_proxy)"
endif
ifdef https_proxy
BUILD_ARGS += --build-arg="https_proxy=$(https_proxy)"
RUN_ARGS += --env="https_proxy=$(https_proxy)"
endif
ifdef APT_PROXY
BUILD_ARGS += --build-arg="APT_PROXY=$(APT_PROXY)"
endif
ifdef PIP_TRUSTED_HOST
BUILD_ARGS += --build-arg="PIP_TRUSTED_HOST=$(PIP_TRUSTED_HOST)"
endif
ifdef PIP_INDEX_URL
BUILD_ARGS += --build-arg="PIP_INDEX_URL=$(PIP_INDEX_URL)"
endif

BUILD_ARGS += --shm-size=1g

GPU := none

help:
	@cat Makefile

.ssh_host_keys:
	mkdir .ssh_host_keys
	ssh-keygen -t ecdsa -f .ssh_host_keys/ssh_host_ecdsa_key -N ''
	ssh-keygen -t ed25519 -f .ssh_host_keys/ssh_host_ed25519_key -N ''
	ssh-keygen -t rsa -f .ssh_host_keys/ssh_host_rsa_key -N ''

rebuild:
	$(MAKE) build BUILD_ARGS="$(BUILD_ARGS) --no-cache"

build: .ssh_host_keys
	docker build --pull $(BUILD_ARGS) --tag=$(IMAGE_TAG) .
	$(MAKE) test
	$(MAKE) test2
	docker images $(IMAGE_TAG)

format:
	docker run --rm --interactive --tty $(RUN_ARGS) \
		--volume="$(CURDIR)/tests:/tests:rw" \
		--user=$(shell id -u) \
		$(IMAGE_TAG) bash -c "isort /tests && black /tests"

test:
	docker run --gpus='"device=$(GPU)"' --rm --interactive --tty $(RUN_ARGS) \
		--volume="$(CURDIR)/tests:/tests:ro" \
		$(IMAGE_TAG) bash -c "pip freeze && pytest /tests"

test2:
	docker run --gpus='"device=$(GPU)"' --rm --interactive --tty $(RUN_ARGS) \
		$(IMAGE_TAG) bash -c "git clone --recursive --depth=1 https://github.com/ak110/pytoolkit.git && cd pytoolkit && pytest"

shell:
	docker run --gpus=all --rm --interactive --tty $(RUN_ARGS) $(IMAGE_TAG) bash

base-shell:
	docker run --gpus=all --rm --interactive --tty $(RUN_ARGS) $(shell head -n 1 Dockerfile | awk '{print$$2}') bash

lint:
	docker pull hadolint/hadolint
	docker run --rm -i hadolint/hadolint < Dockerfile
