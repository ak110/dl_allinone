FROM nvidia/cuda:10.0-cudnn7-devel
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
        software-properties-common \
        wget \
        && \
    wget -q https://www.ubuntulinux.jp/ubuntu-ja-archive-keyring.gpg -O- | apt-key add - && \
    wget -q https://www.ubuntulinux.jp/ubuntu-jp-ppa-keyring.gpg -O- | apt-key add - && \
    wget -q https://www.ubuntulinux.jp/sources.list.d/$DISTRIB_CODENAME.list -O /etc/apt/sources.list.d/ubuntu-ja.list && \
    add-apt-repository ppa:git-core/ppa && \
    wget -q https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh -O- | bash && \
    http_proxy=$APT_PROXY apt-get install --yes --no-install-recommends git git-lfs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# aptその2
RUN set -x && \
    http_proxy=$APT_PROXY apt-get update && \
    http_proxy=$APT_PROXY apt-get install --yes --no-install-recommends \
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
        libbz2-dev \
        libffi-dev \
        libgdbm-dev \
        libjpeg-dev \
        liblzma-dev \
        libmecab-dev \
        libncurses5-dev \
        libopencv-dev \
        libpng-dev \
        libreadline-dev \
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
        tesseract-ocr-jpn \
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
    update-locale LANG=ja_JP.UTF-8 LANGUAGE='ja_JP:ja' && \
    apt-get install --yes --no-install-recommends nodejs npm && \
    npm cache clean && \
    npm install n -g && \
    n stable && \
    apt-get autoremove --purge --yes nodejs npm && \
    apt-get clean
# workaround
RUN set -x && \
    http_proxy=$APT_PROXY apt-get install --yes --no-install-recommends libssl-dev

# OpenMPI
# https://github.com/uber/horovod/blob/master/Dockerfile
# https://www.open-mpi.org/software/
RUN set -x && \
    wget -q https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.0.tar.bz2 -O /opt/openmpi.tar.bz2 && \
    echo "e3da67df1e968c8798827e0e5fe9a510 */opt/openmpi.tar.bz2" | md5sum -c - && \
    cd /opt && \
    tar xfj openmpi.tar.bz2 && \
    cd openmpi-4.0.0 && \
    ./configure --with-cuda --disable-mpi-fortran --disable-java --enable-orterun-prefix-by-default && \
    make -j$(nproc) all && \
    make -j$(nproc) install && \
    ldconfig && \
    rm /opt/openmpi.tar.bz2

# python
# https://github.com/docker-library/python/blob/master/3.7/stretch/Dockerfile
ARG GPG_KEY="0D96DF4D4110E5C43FBFB17F2D347EA6AA65421D"
ARG PYTHON_VERSION="3.6.8"
RUN set -ex \
	\
	&& export GNUPGHOME="$(mktemp -d)" \
	&& echo "==================================================" \
	&& echo workaround of: gpg --batch --keyserver ha.pool.sks-keyservers.net --recv-keys "$GPG_KEY" \
	&& echo "--------------------------------------------------" \
	&& wget --no-cache -O "$GNUPGHOME/key.asc" "http://ha.pool.sks-keyservers.net/pks/lookup?op=get&search=0x2D347EA6AA65421D" \
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
ARG PIP_TRUSTED_HOST=""
ARG PIP_INDEX_URL=""

# pip
ARG PYTHON_PIP_VERSION="19.0.3"
RUN set -ex; \
	\
	wget -O get-pip.py 'https://bootstrap.pypa.io/get-pip.py'; \
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
			\( -type d -a \( -name test -o -name tests \) \) \
			-o \
			\( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
		\) -exec rm -rf '{}' +; \
	rm -f get-pip.py

# 参考: https://github.com/uber/horovod/blob/master/Dockerfile

# https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/tools/pip_package/setup.py : numpy >= 1.14.5, <= 2

# albumentations 0.1.12 : imgaug<0.2.7,>=0.2.5

ARG TENSORFLOW_VERSION=1.13.1
ARG KERAS_VERSION=2.2.4
ARG PYTORCH_VERSION=1.0.1
RUN set -x && \
    pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir \
        "http://download.pytorch.org/whl/cu100/torch-${PYTORCH_VERSION}-cp36-cp36m-linux_x86_64.whl" \
        'git+https://www.github.com/keras-team/keras-contrib.git' \
        'numpy<1.17' \
        'scikit-optimize[plots]' \
        Augmentor \
        Flask \
        Flask-Login \
        Flask-Migrate \
        Flask-Restless \
        Flask-SQLAlchemy \
        Keras==$KERAS_VERSION \
        Pillow \
        albumentations \
        allennlp \
        autopep8 \
        backtrace \
        bandit \
        bcrypt \
        better_exceptions \
        bokeh \
        catboost \
        chainer \
        chainercv \
        chainerrl \
        cnn_finetune \
        cupy-cuda100 \
        cython \
        diskcache \
        fastai \
        fasteners \
        fastprogress \
        fastrlock \
        fire \
        flake8 \
        flake8-docstrings \
        gensim \
        gunicorn \
        h5py \
        hyperopt \
        image-classifiers \
        imageio \
        imbalanced-learn \
        imgaug==0.2.6 \
        imgcrop \
        ipywidgets \
        janome \
        japanize-matplotlib \
        jupyterlab \
        kaggle \
        keras-rl \
        lightgbm \
        matplotlib \
        mecab-python3 \
        mkl \
        mpi4py \
        nltk \
        nose \
        numba \
        opencv-python \
        openpyxl \
        optuna \
        pandas \
        pandas-profiling \
        passlib \
        pep8 \
        pip-tools \
        pipdeptree \
        pipenv \
        pretrainedmodels \
        pycodestyle \
        pydot \
        pygments \
        pylint \
        pyod \
        pypandoc \
        pytesseract \
        pytest \
        pytest-timeout \
        pytest-xdist \
        pytext-nlp \
        python-dotenv \
        pyyaml \
        recommonmark \
        requests \
        rgf_python \
        scikit-image \
        scikit-learn \
        segmentation-models \
        signate \
        six \
        sklearn_pandas \
        spacy \
        sphinx \
        sphinx-autobuild \
        sphinx_rtd_theme \
        tabulate \
        tensorflow-gpu==$TENSORFLOW_VERSION \
        tensorflow-hub \
        tensorpack \
        torchvision \
        tqdm \
        tsfresh \
        xgboost \
        xlrd \
        xlwt \
        xonsh \
        yapf \
        && \
    jupyter nbextension enable --py widgetsnbextension --sys-prefix && \
    jupyter serverextension enable --py jupyterlab --sys-prefix && \
    jupyter labextension install @jupyter-widgets/jupyterlab-manager

# 依存関係の問題があって後回しなやつ
RUN set -x && \
    pip install --no-cache-dir \
        'git+https://github.com/cocodataset/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI' \
        nagisa \
        ptk \
        tslearn \
        ;

# horovod
RUN set -x && \
    ldconfig /usr/local/cuda/lib64/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod && \
    ldconfig

# 最後にPillow-SIMD
RUN set -x && \
    pip uninstall --no-cache-dir --yes Pillow && \
    pip install --no-cache-dir Pillow-SIMD

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
