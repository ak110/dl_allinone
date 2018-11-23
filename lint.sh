#!/bin/bash
set -eux
docker pull hadolint/hadolint
docker run --rm -i hadolint/hadolint < Dockerfile

