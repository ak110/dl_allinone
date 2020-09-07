FROM nvidia/cuda:10.1-cudnn7-devel
ARG DISTRIB_CODENAME=bionic

# 実行時に残さないようにENVではなくARGでnoninteractive
ARG DEBIAN_FRONTEND=noninteractive

# apt用プロキシ(apt-cacher-ng用)
ARG APT_PROXY=$http_proxy

# apt
RUN set -x && \
    sed -ie 's@http://archive.ubuntu.com/ubuntu/@http://ftp.riken.go.jp/Linux/ubuntu/@g' /etc/apt/sources.list && \
    sed -ie 's@^deb-src@# deb-src@g' /etc/apt/sources.list && \
    http_proxy=$APT_PROXY apt-get update && \
    http_proxy=$APT_PROXY apt-get install --yes --no-install-recommends \
        apt-transport-https \
        apt-utils \
        build-essential \
        ca-certificates \
        curl \
        locales \
        software-properties-common \
        wget \
        && \
    locale-gen ja_JP.UTF-8 && \
    update-locale LANG=ja_JP.UTF-8 LANGUAGE='ja_JP:ja' && \
    add-apt-repository ppa:git-core/ppa && \
    wget -q https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh -O- | bash && \
    http_proxy=$APT_PROXY apt-get install --yes --no-install-recommends git git-lfs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV LANG='ja_JP.UTF-8' \
    LC_ALL='ja_JP.UTF-8' \
    LANGUAGE='ja_JP:ja'

# https://horovod.readthedocs.io/en/latest/index.html
# > If you've installed TensorFlow from PyPI, make sure that the g++-4.8.5 or g++-4.9 is installed.
ARG GPP_VERSION=4.8
# aptその2
RUN set -x && \
    yes | env http_proxy=$APT_PROXY unminimize && \
    http_proxy=$APT_PROXY apt-get update && \
    http_proxy=$APT_PROXY apt-get install --yes --no-install-recommends \
        ack-grep \
        apt-file \
        automake \
        bc \
        cifs-utils \
        cmake \
        dialog \
        dpkg-dev \
        emacs \
        entr \
        fonts-ipafont \
        fonts-liberation \
        g++-$GPP_VERSION \
        gdb \
        # for building scipy
        gfortran \
        graphviz \
        hdf5-tools \
        htop \
        iftop \
        imagemagick \
        inetutils-traceroute \
        iotop \
        jq \
        language-pack-ja \
        libatlas-base-dev \
        # for python
        libbluetooth-dev \
        libboost-dev \
        libboost-filesystem-dev \
        libboost-system-dev \
        libbz2-dev \
        libffi-dev \
        libgdbm-dev \
        libjpeg-dev \
        liblzma-dev \
        libmecab-dev \
        libncurses5-dev \
        # TensorRT6 (TensorFlow用)
        libnvinfer-plugin6=6.0.1-1+cuda10.1 \
        libnvinfer6=6.0.1-1+cuda10.1 \
        libopencv-dev \
        libpng-dev \
        libprotobuf-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        libtool \
        libwebp-dev \
        libyaml-dev \
        mecab \
        mecab-ipadic-utf8 \
        mecab-jumandic-utf8 \
        net-tools \
        nvidia-opencl-dev \
        openssh-server \
        p7zip-full \
        pandoc \
        protobuf-compiler \
        screen \
        sl \
        smbclient \
        subversion \
        swig \
        tcl-dev \
        tesseract-ocr \
        tesseract-ocr-jpn \
        tesseract-ocr-jpn-vert \
        tesseract-ocr-script-jpan \
        tesseract-ocr-script-jpan-vert \
        texlive-fonts-recommended \
        texlive-generic-recommended \
        texlive-xetex \
        tk-dev \
        tmux \
        tmuxinator \
        unzip \
        uuid-dev \
        valgrind \
        vim \
        whiptail \
        zip \
        zlib1g-dev \
        zsh \
        && \
    # MeCabの標準はIPA辞書にしておく
    update-alternatives --set mecab-dictionary /var/lib/mecab/dic/ipadic-utf8 && \
    # 後始末
    apt-get clean && \
    # assert
    ls /usr/include/cublas_v2.h > /dev/null && \
    :

# MKL, IPP
# https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-apt-repo
RUN set -x && \
    wget -q https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB -O- | apt-key add - && \
    wget -q https://apt.repos.intel.com/setup/intelproducts.list -O /etc/apt/sources.list.d/intelproducts.list && \
    http_proxy=$APT_PROXY apt-get update && \
    http_proxy=$APT_PROXY apt-get install --yes intel-mkl-64bit-2020.0-088 intel-ipp-64bit-2020.0-088 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "/opt/intel/mkl/lib/intel64" >> /etc/ld.so.conf.d/intel.conf && \
    ldconfig && \
    echo ". /opt/intel/bin/compilervars.sh intel64" > /etc/profile.d/intel.sh

# OpenMPI
# https://github.com/horovod/horovod/blob/master/Dockerfile.gpu
# https://www.open-mpi.org/software/
RUN set -x && \
    mkdir /tmp/openmpi &&\
    cd /tmp/openmpi && \
    wget -q "https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.5.tar.bz2" -O openmpi.tar.bz2 && \
    echo "d85ad67fb1d5decd78a4bb883874e289 *openmpi.tar.bz2" | md5sum -c - && \
    tar xfj openmpi.tar.bz2 && \
    cd openmpi-4.0.5 && \
    ./configure --with-cuda --disable-mpi-fortran --disable-java --enable-orterun-prefix-by-default && \
    make -j$(nproc) all && \
    make -j$(nproc) install && \
    make -j$(nproc) distclean && \
    ldconfig && \
    rm -rf /tmp/openmpi

# python
# https://github.com/docker-library/python/blob/master/3.8/buster/Dockerfile
ARG GPG_KEY="E3FF2839C048B25C084DEBE9B26995E310250568"
ARG GPG_KEY_ID="0xB26995E310250568"
ARG PYTHON_VERSION="3.8.3"
RUN set -ex \
	\
	&& export GNUPGHOME="$(mktemp -d)" \
	&& echo "==================================================" \
	&& echo workaround of: gpg --batch --keyserver ha.pool.sks-keyservers.net --recv-keys "$GPG_KEY" \
	&& echo "--------------------------------------------------" \
	&& wget --no-cache -O "$GNUPGHOME/key.asc" "http://ha.pool.sks-keyservers.net/pks/lookup?op=get&search=$GPG_KEY_ID" \
	&& gpg --import "$GNUPGHOME/key.asc" \
	&& echo "==================================================" \
	&& wget -O python.tar.xz "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz" \
	&& wget -O python.tar.xz.asc "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz.asc" \
	&& gpg --batch --verify python.tar.xz.asc python.tar.xz \
	&& { command -v gpgconf > /dev/null && gpgconf --kill all || :; } \
	&& rm -rf "$GNUPGHOME" python.tar.xz.asc \
	&& mkdir -p /usr/src/python \
	&& tar -xJC /usr/src/python --strip-components=1 -f python.tar.xz \
	&& rm python.tar.xz \
	\
	&& cd /usr/src/python \
	&& gnuArch="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)" \
	&& ./configure \
		--build="$gnuArch" \
		--enable-loadable-sqlite-extensions \
		--enable-optimizations \
		--enable-option-checking=fatal \
		--enable-shared \
		--with-system-expat \
		--with-system-ffi \
		--without-ensurepip \
	&& make -j "$(nproc)" \
	&& make install \
	&& ldconfig \
	\
	&& find /usr/local -depth \
		\( \
			\( -type d -a \( -name test -o -name tests -o -name idle_test \) \) \
			-o \
			\( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
		\) -exec rm -rf '{}' + \
	&& rm -rf /usr/src/python \
	\
	&& python3 --version

# make some useful symlinks that are expected to exist
RUN cd /usr/local/bin \
	&& ln -s idle3 idle \
	&& ln -s pydoc3 pydoc \
	&& ln -s python3 python \
	&& ln -s python3-config python-config

# devpi-server用
ARG PIP_TRUSTED_HOST=""
ARG PIP_INDEX_URL=""

ARG PYTHON_PIP_VERSION="20.1.1"
ARG PYTHON_GET_PIP_URL="https://github.com/pypa/get-pip/raw/eff16c878c7fd6b688b9b4c4267695cf1a0bf01b/get-pip.py"
ARG PYTHON_GET_PIP_SHA256="b3153ec0cf7b7bbf9556932aa37e4981c35dc2a2c501d70d91d2795aa532be79"

RUN set -ex; \
	\
	wget -O get-pip.py "$PYTHON_GET_PIP_URL"; \
	echo "$PYTHON_GET_PIP_SHA256 *get-pip.py" | sha256sum --check --strict -; \
	\
	python get-pip.py \
		--disable-pip-version-check \
		--no-cache-dir \
		"pip==$PYTHON_PIP_VERSION" \
	; \
	pip --version; \
	\
	find /usr/local -depth \
		\( \
			\( -type d -a \( -name test -o -name tests -o -name idle_test \) \) \
			-o \
			\( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
		\) -exec rm -rf '{}' +; \
	rm -f get-pip.py

# numpy/scipy with MKL
RUN set -x && \
    echo '[mkl]' > /root/.numpy-site.cfg && \
    echo 'library_dirs = /opt/intel/mkl/lib/intel64' >> /root/.numpy-site.cfg && \
    echo 'include_dirs = /opt/intel/mkl/include' >> /root/.numpy-site.cfg && \
    echo 'mkl_libs = mkl_rt' >> /root/.numpy-site.cfg && \
    echo 'lapack_libs =' >> /root/.numpy-site.cfg && \
    pip install --no-binary :all: numpy\<1.18 scipy

RUN set -x && \
    pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir \
        Augmentor \
        Flask \
        Flask-Login \
        Flask-Migrate \
        Flask-Restless \
        Flask-SQLAlchemy \
        Keras \
        Pillow \
        albumentations \
        autopep8 \
        bandit \
        bashplotlib \
        bcrypt \
        better_exceptions \
        black \
        bokeh \
        cairocffi \
        catboost \
        category_encoders \
        chainer \
        chainerrl \
        cookiecutter \
        cupy-cuda101 \
        cysignals \
        cython \
        diskcache \
        editdistance \
        efficientnet\>=1.1.0 \
        eli5 \
        ensemble-boxes \
        fasteners \
        fastprogress \
        fastrlock \
        feather-format \
        featuretools \
        fire \
        flake8 \
        flake8-bugbear \
        gensim \
        gluoncv \
        gluonnlp \
        gunicorn \
        hyperopt \
        image-classifiers \
        imageio \
        imbalanced-learn \
        # albumentations : imgaug<0.2.7,>=0.2.5  <https://github.com/albumentations-team/albumentations/blob/master/setup.py#L9>
        imgaug==0.2.6 \
        imgdup \
        ipadic \
        ipywidgets \
        iterative-stratification \
        janome \
        japanize-matplotlib \
        kaggle \
        keras2onnx \
        matplotlib \
        mecab-python3 \
        mpi4py \
        mxnet-cu101mkl \
        mypy \
        nlp \
        nltk \
        noise \
        nose \
        numba \
        onnxmltools \
        opencv-python-headless \
        openpyxl \
        optuna\>=1.3.0 \
        pandas \
        pandas-profiling \
        passlib \
        pip-tools \
        pipdeptree \
        pipenv \
        plotly \
        poetry \
        pre-commit \
        progressbar2 \
        pycryptodome \
        # nlp
        pyarrow\>=0.16.0,\<1.0.0 \
        pydot \
        pygments \
        pylama \
        pylama_pylint \
        pylint \
        pyod \
        pypandoc \
        pytesseract \
        pytest \
        pytest-cookies \
        pytest-timeout \
        pytest-xdist \
        python-dotenv \
        python-utils \
        pyupgrade \
        pyyaml \
        recommonmark \
        requests \
        rgf_python \
        rope \
        safety \
        scikit-image \
        scikit-learn \
        scikit-optimize\[plots\] \
        seaborn \
        segmentation-models \
        setup-cfg-fmt \
        signate \
        six \
        sklearn_pandas \
        spacy \
        sphinx \
        sphinx-autobuild \
        sphinx-autodoc-typehints \
        sphinx_rtd_theme \
        stickytape \
        sympy \
        tabulate \
        tensorflow-addons\>=0.10.0 \
        tensorflow-datasets \
        tensorflow-hub \
        tensorflow==2.3.0 \
        tensorpack \
        # https://github.com/explosion/spaCy/issues/2883
        # https://github.com/explosion/spaCy/blob/master/requirements.txt
        thinc==7.4.1 \
        tf2cv \
        tf2onnx \
        tokenizers \
        tqdm \
        transformers \
        tsfresh \
        # mypy用バージョン指定。なぜかchainerのPython2用の依存関係に従ってしまう？
        typing-extensions\>=3.7.4 \
        unidic-lite \
        xgboost \
        xlrd \
        xlwt \
        xonsh \
        yapf \
        # 依存関係に注意
        chainercv

# PyTorch関連
RUN set -x && \
    # PyTorchが既にインストールされてしまっていないことの確認
    test $(pip freeze | grep ^torch== | wc -l) -eq 0 && \
    # PyTorchとそれに依存するものたちのインストール
    pip install --no-cache-dir \
        allennlp \
        cnn-finetune \
        fastai \
        pretrainedmodels \
        pytorch-ignite \
        pytorch-lightning \
        torch==1.6.0+cu101 \
        torchaudio \
        torchtext \
        torchvision==0.6.1+cu101 \
        --find-links=https://download.pytorch.org/whl/torch_stable.html

# apex
RUN set -x && \
    git clone --depth=1 https://github.com/NVIDIA/apex.git /tmp/apex &&\
    cd /tmp/apex &&\
    pip install --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ &&\
    rm -rf /tmp/apex

# 依存関係の問題があって後回しなやつ
RUN set -x && \
    pip install --no-cache-dir \
        fasttext \
        lycon \
        ptk \
        pycocotools \
        tslearn \
        ;

# 辞書など
RUN set -x && \
    python3 -m nltk.downloader -d /usr/local/share/nltk_data popular
RUN set -x && \
    python3 -m spacy download en --no-cache

# nodejs
ARG NODEJS_VERSION=v12.18.3
RUN set -x && \
    wget -q -O- https://nodejs.org/dist/$NODEJS_VERSION/node-$NODEJS_VERSION-linux-x64.tar.xz | tar xJ -C /tmp/ && \
    mv /tmp/node-$NODEJS_VERSION-linux-x64/bin/* /usr/local/bin/ && \
    mv /tmp/node-$NODEJS_VERSION-linux-x64/lib/* /usr/local/lib/ && \
    mv /tmp/node-$NODEJS_VERSION-linux-x64/include/* /usr/local/include/ && \
    mv /tmp/node-$NODEJS_VERSION-linux-x64/share/doc/* /usr/local/share/doc/ && \
    mv /tmp/node-$NODEJS_VERSION-linux-x64/share/man/man1/* /usr/local/share/man/man1 && \
    rm -rf /tmp/node-$NODEJS_VERSION-linux-x64

# jupyter関連
# plotly: https://plot.ly/python/getting-started/#jupyterlab-support-python-35
RUN set -x && \
    pip install --no-cache-dir \
        jupyter-tensorboard \
        jupyterlab \
        jupyterlab-git \
        && \
    # export NODE_OPTIONS=--max-old-space-size=4096 &&\
    # (jupyter labextension install --debug-log-path=/tmp/jupyterlab-build.log \
    #     @jupyter-widgets/jupyterlab-manager \
    #     @jupyterlab/git \
    #     @jupyterlab/toc \
    #     jupyterlab-chart-editor \
    #     jupyterlab-plotly \
    #     jupyterlab_tensorboard \
    #     plotlywidget \
    #     || (cat /tmp/jupyterlab-build.log && false)) && \
    jupyter serverextension enable --sys-prefix --py \
        jupyterlab \
        jupyterlab_git

# LightGBM
RUN set -x && \
    pip install --no-cache-dir --install-option=--gpu lightgbm

# horovod
# 参考: https://github.com/horovod/horovod/blob/master/Dockerfile.gpu
RUN set -x && \
    ldconfig /usr/local/cuda/lib64/stubs && \
    HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=1 \
        pip install --no-cache-dir horovod && \
    ldconfig

# 最後にPillow-SIMD
RUN set -x && \
    CC="cc -mavx2" pip install --no-cache-dir --force-reinstall Pillow-SIMD

# npm
RUN set -x && \
    npm install -g pyright npm-check-updates

# ユーザー作成
ARG RUN_USER=user
ARG RUN_UID=1000
RUN set -x && \
    useradd --create-home --shell=/bin/bash --uid=$RUN_UID $RUN_USER

RUN set -x && \
    # sshd用ディレクトリ作成
    mkdir --mode=744 /var/run/sshd && \
    # sshdで~/.ssh/environmentがあれば読み込むようにする
    sed -i 's/#PermitUserEnvironment no/PermitUserEnvironment yes/' /etc/ssh/sshd_config && \
    # horovod用のNCCL設定
    echo 'NCCL_DEBUG=INFO' >> /etc/nccl.conf && \
    # horovod用のOpenMPI設定
    echo 'hwloc_base_binding_policy = none' >> /usr/local/etc/openmpi-mca-params.conf && \
    echo 'rmaps_base_mapping_policy = slot' >> /usr/local/etc/openmpi-mca-params.conf && \
    echo 'btl_tcp_if_exclude = lo,docker0' >> /usr/local/etc/openmpi-mca-params.conf && \
    # 環境変数設定
    echo 'export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH' > /etc/profile.d/docker.sh && \
    echo 'export BETTER_EXCEPTIONS=1' >> /etc/profile.d/docker.sh && \
    echo 'export TF_FORCE_GPU_ALLOW_GROWTH=true' >> /etc/profile.d/docker.sh && \
    # sudoでhttp_proxyなどが引き継がれるようにしておく
    echo 'Defaults env_keep += "http_proxy https_proxy ftp_proxy no_proxy PIP_TRUSTED_HOST PIP_INDEX_URL"' > /etc/sudoers.d/docker && \
    echo 'Defaults always_set_home' >> /etc/sudoers.d/docker && \
    # $RUN_USERをパスワード無しでsudoできるようにしておく
    echo $RUN_USER ALL=\(ALL\) NOPASSWD:ALL >> /etc/sudoers.d/docker　&&\
    chmod 0440 /etc/sudoers.d/* && \
    visudo --check && \
    # completion
    poetry completions bash > /etc/bash_completion.d/poetry.bash-completion && \
    # 最後にldconfigしておく
    ldconfig

# sshd以外の使い方をするとき用環境変数色々
ENV TZ='Asia/Tokyo' \
    PYTHONIOENCODING='utf-8' \
    PYTHONDONTWRITEBYTECODE=1 \
    BETTER_EXCEPTIONS=1 \
    TF_FORCE_GPU_ALLOW_GROWTH='true'

# SSHホストキーを固定で用意
COPY --chown=root:root .ssh_host_keys/ssh_host_* /etc/ssh/
# 作った日時を記録しておく (一応)
RUN date '+%Y/%m/%d %H:%M:%S' > /image.version
# sshd
# -D: デタッチしない
# -e: ログを標準エラーへ
CMD ["/usr/sbin/sshd", "-D", "-e"]
