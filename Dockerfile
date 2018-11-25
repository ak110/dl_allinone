FROM nvidia/cuda:9.0-cudnn7-devel

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
        software-properties-common \
        wget \
        && \
    wget -q https://www.ubuntulinux.jp/ubuntu-ja-archive-keyring.gpg -O- | apt-key add - && \
    wget -q https://www.ubuntulinux.jp/ubuntu-jp-ppa-keyring.gpg -O- | apt-key add - && \
    wget -q https://www.ubuntulinux.jp/sources.list.d/xenial.list -O /etc/apt/sources.list.d/ubuntu-ja.list && \
    add-apt-repository ppa:git-core/ppa && \
    wget -q https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh -O- | bash && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# aptその2
RUN set -x && \
    http_proxy=$APT_PROXY apt-get update && \
    http_proxy=$APT_PROXY apt-get install --yes --no-install-recommends --allow-downgrades \
        ack-grep \
        apt-file \
        automake \
        bash-completion \
        bc \
        bsdmainutils \
        cifs-utils \
        cmake \
        command-not-found \
        cpio \
        debconf-i18n \
        dialog \
        dpkg-dev \
        ed \
        emacs \
        entr \
        file \
        fonts-ipafont \
        fonts-liberation \
        g++ \
        gdb \
        git \
        git-lfs \
        graphviz \
        hdf5-tools \
        htop \
        iftop \
        imagemagick \
        inetutils-traceroute \
        iotop \
        iproute2 \
        iputils-ping \
        jq \
        language-pack-ja \
        less \
        libatlas-base-dev \
        libboost-all-dev \
        libbz2-dev \
        libgdbm-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libjpeg-dev \
        libleveldb-dev \
        liblmdb-dev \
        liblzma-dev \
        libmecab-dev \
        libncurses5-dev \
        libopencv-dev \
        libpng-dev \
        libprotobuf-dev \
        libreadline-dev \
        libsnappy-dev \
        libsqlite3-dev \
        libssl-dev \
        libtool \
        man-db \
        mecab \
        mecab-ipadic-utf8 \
        net-tools \
        openssh-client \
        openssh-server \
        p7zip-full \
        pandoc \
        protobuf-compiler \
        psmisc \
        rsync \
        screen \
        sl \
        smbclient \
        subversion \
        sudo \
        swig \
        tcl-dev \
        telnet \
        tk-dev \
        tmux \
        tmuxinator \
        unzip \
        valgrind \
        vim \
        whiptail \
        zip \
        zlib1g-dev \
        zsh \
        && \
    apt-get install --yes --no-install-recommends nodejs npm && \
    npm cache clean && \
    npm install n -g && \
    n stable && \
    apt-get purge --yes nodejs npm && \
    update-locale LANG=ja_JP.UTF-8 LANGUAGE='ja_JP:ja' && \
    apt-get clean

# OpenMPI
# 参考：https://github.com/uber/horovod/blob/master/Dockerfile
RUN set -x && \
    wget -q https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.2.tar.bz2 -O /opt/openmpi.tar.bz2 && \
    echo "210df69fafd964158527e7f37e333239 */opt/openmpi.tar.bz2" | md5sum -c - && \
    cd /opt && \
    tar xfj openmpi.tar.bz2 && \
    cd openmpi-3.1.2 && \
    ./configure --with-cuda --disable-mpi-fortran --disable-java --enable-orterun-prefix-by-default && \
    make -j$(nproc) all && \
    make -j$(nproc) install && \
    ldconfig && \
    rm /opt/openmpi.tar.bz2

# python
# https://github.com/docker-library/python/blob/master/3.6/stretch/Dockerfile
ARG GPG_KEY="0D96DF4D4110E5C43FBFB17F2D347EA6AA65421D"
ARG PYTHON_VERSION="3.6.7"
RUN set -ex \
	&& wget -O python.tar.xz "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz" \
	&& wget -O python.tar.xz.asc "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz.asc" \
	&& export GNUPGHOME="$(mktemp -d)" \
	&& gpg --batch --keyserver ha.pool.sks-keyservers.net --recv-keys "$GPG_KEY" \
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
			\( -type d -a \( -name test -o -name tests \) \) \
			-o \
			\( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
		\) -exec rm -rf '{}' + \
	&& rm -rf /usr/src/python \
    && cd /usr/local/bin \
    && ln -s idle3 idle \
    && ln -s pydoc3 pydoc \
    && ln -s python3 python \
    && ln -s python3-config python-config

# devpi-server用
ARG PIP_PROXY=$http_proxy
ARG PIP_TRUSTED_HOST=""
ARG PIP_INDEX_URL=""

# pip
RUN set -x && \
    wget 'https://bootstrap.pypa.io/get-pip.py' -O get-pip.py && \
    python get-pip.py --no-cache-dir && \
    rm -f get-pip.py
RUN set -x && \
    http_proxy=$PIP_PROXY pip install --upgrade --no-cache-dir pip && \
    http_proxy=$PIP_PROXY pip install --no-cache-dir \
        Pillow-SIMD \
        bcolz \
        cython \
        fastrlock \
        futures==3.1.1 \
        gensim \
        graphviz \
        gym \
        h5py \
        matplotlib \
        mkl \
        mpi4py \
        nose \
        numba \
        numpy \
        pandas \
        pbr \
        pydot \
        pydot_ng \
        pypandoc \
        pyyaml \
        scikit-image \
        scikit-learn \
        scipy \
        setuptools_scm \
        six \
        sklearn_pandas \
        spacy \
        ;

# Chainer
RUN set -x && \
    http_proxy=$PIP_PROXY pip install --no-cache-dir cupy-cuda90 chainer chainercv chainerrl

# PyTorch
ARG PYTORCH_VERSION=0.4.1
RUN set -x && \
    pip install --no-cache-dir http://download.pytorch.org/whl/cu90/torch-${PYTORCH_VERSION}-cp36-cp36m-linux_x86_64.whl && \
    http_proxy=$PIP_PROXY pip install --no-cache-dir \
        torchvision \
        cnn_finetune \
        pretrainedmodels \
        fastai \
        ;

# Keras+TensorFlow
# https://github.com/uber/horovod/blob/master/Dockerfile
ARG TENSORFLOW_VERSION=1.10.0
ARG KERAS_VERSION=2.2.4
RUN http_proxy=$PIP_PROXY pip install --no-cache-dir tensorflow-gpu==$TENSORFLOW_VERSION
RUN http_proxy=$PIP_PROXY pip install --no-cache-dir Keras==$KERAS_VERSION

# horovod
RUN set -x && \
    ldconfig /usr/local/cuda/lib64/stubs && \
    http_proxy=$PIP_PROXY HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod && \
    ldconfig

# その他Pythonライブラリ色々
RUN set -x && \
    http_proxy=$PIP_PROXY pip install --no-cache-dir \
        'git+https://github.com/cocodataset/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI' \
        'git+https://www.github.com/keras-team/keras-contrib.git' \
        'scikit-optimize[plots]' \
        Augmentor \
        Flask \
        Flask-Migrate \
        Flask-SQLAlchemy \
        GPyOpt \
        albumentations \
        autopep8 \
        bandit \
        better_exceptions \
        catboost \
        diskcache \
        fasteners \
        fire \
        flake8 \
        flake8-docstrings \
        flake8-pep257 \
        gunicorn \
        hyperopt \
        imageio \
        imbalanced-learn \
        imgaug \
        imgcrop \
        ipywidgets \
        janome \
        jupyterlab \
        kaggle \
        kaggle-cli \
        keras-rl \
        lightgbm \
        mecab-python3 \
        nagisa \
        nltk \
        opencv-python \
        openpyxl \
        pandas-profiling \
        pip-tools \
        pipdeptree \
        pycodestyle \
        pylint \
        pytest-timeout \
        pytest-xdist \
        pytest==3.9.3 \
        python-dotenv \
        tabulate \
        tensorflow-hub \
        tqdm \
        tslearn \
        xgboost \
        xlrd \
        xlwt \
        yapf \
        && \
    jupyter nbextension enable --py widgetsnbextension --sys-prefix && \
    jupyter serverextension enable --py jupyterlab --sys-prefix && \
    jupyter labextension install @jupyter-widgets/jupyterlab-manager

# monkey patch
COPY sitecustomize.py /usr/local/lib/python3.6/

# ・sshd用ディレクトリ作成
# ・horovod用のNCCL / OpenMPI設定
# ・cuda、pythonなどのパスを通す
# ・matplotlibがエラーにならないようにMPLBACKEND=Aggを設定
# ・sudoでhttp_proxyなどが引き継がれるようにしておく
# ・最後にldconfigしておく
RUN set -x && \
    mkdir --mode=744 /var/run/sshd && \
    echo NCCL_DEBUG=INFO >> /etc/nccl.conf && \
    echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf && \
    echo 'hwloc_base_binding_policy = none' >> /usr/local/etc/openmpi-mca-params.conf && \
    echo 'rmaps_base_mapping_policy = slot' >> /usr/local/etc/openmpi-mca-params.conf && \
    echo 'btl_tcp_if_exclude = lo,docker0' >> /usr/local/etc/openmpi-mca-params.conf && \
    echo 'export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH' > /etc/profile.d/docker.sh && \
    echo 'export MPLBACKEND=Agg' >> /etc/profile.d/docker.sh && \
    echo 'Defaults env_keep += "http_proxy https_proxy ftp_proxy no_proxy"' > /etc/sudoers.d/docker && \
    echo 'Defaults always_set_home' >> /etc/sudoers.d/docker && \
    chmod 0440 /etc/sudoers.d/* && \
    visudo --check && \
    ldconfig

# sshd以外の使い方をするとき用環境変数色々
ENV TZ='Asia/Tokyo' \
    LANG='ja_JP.UTF-8' \
    LC_ALL='ja_JP.UTF-8' \
    LANGUAGE='ja_JP:ja' \
    MPLBACKEND='Agg'

COPY start_sshd.sh /root/
RUN date '+%Y/%m/%d %H:%M:%S' > /image.version
CMD ["/root/start_sshd.sh"]
