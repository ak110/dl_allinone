# `dl_allinone`

良い子は真似してはいけない(?)全部入りDockerイメージ。

## 準備

    cp .env-example .env
    vim .env

## build

    make build

## run

    docker run \
        --gpus=all \
        --detach \
        --restart=always \
        --volume="/etc/localtime:/etc/localtime:ro" \
        --volume="/お好みのパス:/home/user" \
        --volume="/お好みのパス:/data" \
        --env="SSH_USER=user" \
        --env="SSH_UID=1000" \
        --env="SSH_KEY=SSHキー" \
        --publish="お好みのポート番号:22" \
        --name=dl_allinone dl_allinone

環境変数:

- `SSH_USER`: ユーザ名
- `SSH_UID`: UID
- `SSH_KEY`: SSHキー (例: ssh-rsa AAAAB3Nza(略)ILALci+4zLDQ0w==)
- `SSH_PASS`: パスワード

`SSH_KEY` と `SSH_PASS` はどちらか必須。


## logs

    docker logs dl_allinone

## exec

    docker exec -it dl_allinone bash

## rm

    docker rm -f dl_allinone
