
include .env

ifdef http_proxy
BUILD_ARGS += --build-arg="http_proxy=$(http_proxy)"
endif
ifdef https_proxy
BUILD_ARGS += --build-arg="https_proxy=$(https_proxy)"
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

help:
	@cat Makefile

.ssh_host_keys:
	mkdir .ssh_host_keys
	ssh-keygen -t ecdsa -f .ssh_host_keys/ssh_host_ecdsa_key -N ''
	ssh-keygen -t ed25519 -f .ssh_host_keys/ssh_host_ed25519_key -N ''
	ssh-keygen -t rsa -f .ssh_host_keys/ssh_host_rsa_key -N ''

build: .ssh_host_keys
	docker build --pull $(BUILD_ARGS) --tag=$(IMAGE_TAG) .
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
