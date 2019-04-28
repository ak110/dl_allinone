
include .env

help:
	@cat Makefile

build:
	docker build --pull \
		--build-arg http_proxy=$(http_proxy) \
		--build-arg https_proxy=$(https_proxy) \
		--build-arg APT_PROXY=$(APT_PROXY) \
		--build-arg PIP_TRUSTED_HOST=$(PIP_TRUSTED_HOST) \
		--build-arg PIP_INDEX_URL=$(PIP_INDEX_URL) \
		--tag=$(IMAGE_TAG) .
	$(MAKE) test
	$(MAKE) test2
	docker images $(IMAGE_TAG)

test:
	docker run --runtime=nvidia --rm --interactive --tty \
		--volume="$(CURDIR)/tests:/tests:ro" \
		--env="NVIDIA_VISIBLE_DEVICES=$(GPU)" \
		$(IMAGE_TAG) pytest /tests

test2:
	docker run --runtime=nvidia --rm --interactive --tty \
		--env="NVIDIA_VISIBLE_DEVICES=$(GPU)" \
		$(IMAGE_TAG) bash -c "https_proxy=$(https_proxy) git clone --recursive https://github.com/ak110/pytoolkit.git && cd pytoolkit && pytest"

shell:
	docker run --runtime=nvidia --rm --interactive --tty $(IMAGE_TAG) bash

lint:
	docker pull hadolint/hadolint
	docker run --rm -i hadolint/hadolint < Dockerfile
