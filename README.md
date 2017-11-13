# dl_allinone

良い子は真似してはいけない全部入りDockerイメージ。

## build

通常：

    docker build --tag=dl_allinone .

プロキシ環境：

    docker build --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy --tag=dl_allinone .

