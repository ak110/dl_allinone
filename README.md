# dl_allinone

良い子は真似してはいけない(?)全部入りDockerイメージ。

## build

    docker build --build-arg --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy --tag=dl_allinone .

apt-cacher-ngを使う場合： `--build-arg APT_PROXY=http://xxxx:3142`

devpi-serverを使う場合： `--build-arg PIP_PROXY="" --build-arg PIP_TRUSTED_HOST=xxxx --build-arg PIP_INDEX_URL=http://xxxx:3141/root/pypi/`

## run

    NV_GPU=0,1,2,3 nvidia-docker run \
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
