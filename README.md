# `dl_allinone`

良い子は真似してはいけない(?)全部入りDockerイメージ。

## 準備

    cp .env-example .env
    vim .env

## build

    make build

## 動作確認

    make shell

## run (例)

    docker run \
        --gpus=all \
        --detach \
        --restart=always \
        --volume="/etc/localtime:/etc/localtime:ro" \
        --volume="/お好みのパス1:/home/user" \
        --volume="/お好みのパス1:/data" \
        --publish="お好みのポート番号:22" \
        --name=dl_allinone dl_allinone

事前に`/お好みのパス1/.ssh/authorized_keys`を作成しておきSSHログインして使う。
(もしくは実行したいコマンドを付けてdocker runする。)

UIDは(ビルド時に変えていなければ)1000。

## logs

    docker logs dl_allinone

## exec

    docker exec -it dl_allinone bash

## rm

    docker rm -f dl_allinone
