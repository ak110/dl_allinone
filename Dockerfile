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
        net-tools \
        nvidia-opencl-dev \
        openssh-server \
        p7zip-full \
        pandoc \
        protobuf-compiler \
        # for venv
        python3-distutils \
        screen \
        sl \
        smbclient \
        subversion \
        swig \
        tcl-dev \
        tesseract-ocr \
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
    wget -q "https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.0.tar.bz2" -O /opt/openmpi.tar.bz2 && \
    echo "e3da67df1e968c8798827e0e5fe9a510 */opt/openmpi.tar.bz2" | md5sum -c - && \
    cd /opt && \
    tar xfj openmpi.tar.bz2 && \
    cd openmpi-4.0.0 && \
    ./configure --with-cuda --disable-mpi-fortran --disable-java --enable-orterun-prefix-by-default && \
    make -j$(nproc) all && \
    make -j$(nproc) install && \
    make -j$(nproc) distclean && \
    ldconfig && \
    rm /opt/openmpi.tar.bz2

# python
# https://github.com/docker-library/python/blob/master/3.7/stretch/Dockerfile
ARG GPG_KEY="0D96DF4D4110E5C43FBFB17F2D347EA6AA65421D"
ARG PYTHON_VERSION="3.7.6"
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
		--enable-optimizations \
		--enable-option-checking=fatal \
		--enable-shared \
		--with-system-expat \
		--with-system-ffi \
		--without-ensurepip \
	&& make -j "$(nproc)" \
# setting PROFILE_TASK makes "--enable-optimizations" reasonable: https://bugs.python.org/issue36044 / https://github.com/docker-library/python/issues/160#issuecomment-509426916
		PROFILE_TASK='-m test.regrtest --pgo \
			test_array \
			test_base64 \
			test_binascii \
			test_binhex \
			test_binop \
			test_bytes \
			test_c_locale_coercion \
			test_class \
			test_cmath \
			test_codecs \
			test_compile \
			test_complex \
			test_csv \
			test_decimal \
			test_dict \
			test_float \
			test_fstring \
			test_hashlib \
			test_io \
			test_iter \
			test_json \
			test_long \
			test_math \
			test_memoryview \
			test_pickle \
			test_re \
			test_set \
			test_slice \
			test_struct \
			test_threading \
			test_time \
			test_traceback \
			test_unicode \
		' \
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

# if this is called "PIP_VERSION", pip explodes with "ValueError: invalid truth value '<VERSION>'"
ARG PYTHON_PIP_VERSION="20.0.2"
# https://github.com/pypa/get-pip
ARG PYTHON_GET_PIP_URL="https://github.com/pypa/get-pip/raw/42ad3426cb1ef05863521d7988d5f7fec0c99560/get-pip.py"
ARG PYTHON_GET_PIP_SHA256="da288fc002d0bb2b90f6fbabc91048c1fa18d567ad067ee713c6e331d3a32b45"

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

ARG TENSORFLOW_VERSION=2.1.0
ARG PYTORCH_VERSION=1.4.0
ARG TORCHVISION_VERSION=0.5.0
RUN set -x && \
    pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir \
        'scikit-optimize[plots]' \
        # mypy用バージョン指定。なぜかchainerのPython2用の依存関係に従ってしまう？
        'typing-extensions>=3.7.4' \
        # 何故かpipenvのリリースが滞っているのでGitHubから。2019/11/10 20:09 JST版。
        'git+https://github.com/pypa/pipenv.git@3e63f07' \
        Augmentor \
        Flask \
        Flask-Login \
        Flask-Migrate \
        Flask-Restless \
        Flask-SQLAlchemy \
        Keras \
        Pillow \
        albumentations \
        allennlp \
        autopep8 \
        bandit \
        bcrypt \
        better_exceptions \
        black \
        bokeh \
        cairocffi \
        catboost \
        category_encoders \
        chainer \
        chainerrl \
        cnn_finetune \
        cookiecutter \
        cupy-cuda101 \
        cysignals \
        cython \
        diskcache \
        editdistance \
        eli5 \
        fastai \
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
        h5py \
        hyperopt \
        image-classifiers \
        imageio \
        imbalanced-learn \
        # albumentations : imgaug<0.2.7,>=0.2.5  <https://github.com/albumentations-team/albumentations/blob/master/setup.py#L9>
        imgaug==0.2.6 \
        imgcrop \
        imgdup \
        ipywidgets \
        iterative-stratification \
        janome \
        japanize-matplotlib \
        jupyterlab \
        jupyterlab-git \
        kaggle \
        keras2onnx \
        matplotlib \
        mecab-python3 \
        mkl \
        mpi4py \
        mxnet-cu101mkl \
        mypy \
        nltk \
        noise \
        nose \
        numba \
        # https://github.com/tensorflow/tensorflow/issues/30120
        # https://github.com/tensorflow/tensorflow/issues/31249
        'numpy<1.17' \
        onnxmltools \
        opencv-python \
        openpyxl \
        optuna \
        pandas \
        pandas-profiling \
        passlib \
        pip-tools \
        pipdeptree \
        plotly \
        poetry \
        pretrainedmodels \
        progressbar2 \
        pycodestyle \
        pydot \
        pygments \
        pylint \
        pyod \
        pypandoc \
        pytesseract \
        pytest \
        pytest-cookies \
        pytest-timeout \
        pytest-xdist \
        pytext-nlp \
        python-dotenv \
        python-utils \
        pytorch-ignite \
        pytorch-lightning \
        pyyaml \
        recommonmark \
        requests \
        rgf_python \
        rope \
        scikit-image \
        scikit-learn \
        seaborn \
        segmentation-models \
        signate \
        six \
        sklearn_pandas \
        spacy \
        sphinx \
        sphinx-autobuild \
        sphinx-autodoc-typehints \
        sphinx_rtd_theme \
        stickytape \
        tabulate \
        tensorflow-addons \
        tensorflow-datasets \
        tensorflow-hub \
        tensorflow==$TENSORFLOW_VERSION \
        tensorpack \
        tf2cv \
        tf2onnx \
        torch \
        torchaudio \
        torchtext \
        torchvision \
        tqdm \
        tsfresh \
        xgboost \
        xlrd \
        xlwt \
        xonsh \
        yapf \
        # 依存関係に注意
        chainercv \
        jupyter-tensorboard \
        && \
    python3 -m nltk.downloader -d /usr/local/share/nltk_data popular && \
    python3 -m spacy download en --no-cache

# 依存関係の問題があって後回しなやつ
RUN set -x && \
    pip install --no-cache-dir \
        fasttext \
        ptk \
        pycocotools \
        tslearn \
        ;

# jupyter関連
# plotly: https://plot.ly/python/getting-started/#jupyterlab-support-python-35
RUN set -x && \
    export NODE_OPTIONS=--max-old-space-size=4096 &&\
    (jupyter labextension install --debug-log-path=/tmp/jupyterlab-build.log \
        @jupyter-widgets/jupyterlab-manager \
        @jupyterlab/git \
        @jupyterlab/toc \
        # jupyterlab-chart-editor \
        # jupyterlab-plotly \
        jupyterlab_tensorboard \
        # plotlywidget \
        || (cat /tmp/jupyterlab-build.log && false)) && \
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
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=1 \
        pip install --no-cache-dir horovod && \
    ldconfig

# 最後にPillow-SIMD
RUN set -x && \
    pip uninstall --no-cache-dir --yes Pillow && \
    pip install --no-cache-dir 'Pillow-SIMD<7.0.0'

# pyright
RUN set -x && \
    npm install -g pyright

# ・sshd用ディレクトリ作成
# ・sshdでPermitUserEnvironment yes
# ・horovod用のNCCL / OpenMPI設定
# ・cuda、pythonなどのパスを通す
# ・sudoでhttp_proxyなどが引き継がれるようにしておく
# ・最後にldconfigしておく
RUN set -x && \
    mkdir --mode=744 /var/run/sshd && \
    sed -i 's/#PermitUserEnvironment no/PermitUserEnvironment yes/' /etc/ssh/sshd_config && \
    echo 'NCCL_DEBUG=INFO' >> /etc/nccl.conf && \
    echo 'hwloc_base_binding_policy = none' >> /usr/local/etc/openmpi-mca-params.conf && \
    echo 'rmaps_base_mapping_policy = slot' >> /usr/local/etc/openmpi-mca-params.conf && \
    echo 'btl_tcp_if_exclude = lo,docker0' >> /usr/local/etc/openmpi-mca-params.conf && \
    echo 'export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH' > /etc/profile.d/docker.sh && \
    echo 'export TF_FORCE_GPU_ALLOW_GROWTH=true' >> /etc/profile.d/docker.sh && \
    echo 'Defaults env_keep += "http_proxy https_proxy ftp_proxy no_proxy"' > /etc/sudoers.d/docker && \
    echo 'Defaults always_set_home' >> /etc/sudoers.d/docker && \
    chmod 0440 /etc/sudoers.d/* && \
    visudo --check && \
    ldconfig

# sshd以外の使い方をするとき用環境変数色々
ENV TZ='Asia/Tokyo' \
    TF_FORCE_GPU_ALLOW_GROWTH='true' \
    PYTHONDONTWRITEBYTECODE=1

COPY --chown=root:root .ssh_host_keys/ssh_host_* /etc/ssh/
COPY start_sshd.sh /root/
RUN date '+%Y/%m/%d %H:%M:%S' > /image.version
CMD ["/root/start_sshd.sh"]
