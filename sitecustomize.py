"""monkey patch。

■TensorFlowで強制的にconfig.gpu_options.allow_growth = Trueをする
https://qiita.com/ak11/items/875c0f520ff1e231ee0c

"""
import importlib.machinery
import warnings
import sys


class CustomFinder(importlib.machinery.PathFinder):
    """tf.Sessionのimport時にCustomLoaderを使用するFinder。"""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'tensorflow.python.client.session':
            spec = super().find_spec(fullname, path, target)
            loader = CustomLoader(fullname, spec.origin)
            return importlib.machinery.ModuleSpec(fullname, loader)
        return None


class CustomLoader(importlib.machinery.SourceFileLoader):
    """tf.Session.__init__に`config.gpu_options.allow_growth`を強制的にTrueにする処理を入れ込むLoader。"""

    def exec_module(self, module):
        r = super().exec_module(module)
        self._patch(module)
        return r

    def _patch(self, module):
        original_init = module.Session.__init__

        def custom_init(self, *args, config=None, **kwargs):
            import tensorflow as tf
            if config is None:
                config = tf.ConfigProto()
            if not config.gpu_options.allow_growth:
                warnings.warn('Invalid session config: `gpu_options.allow_growth` should be True.')
                config.gpu_options.allow_growth = True
            return original_init(self, *args, config=config, **kwargs)

        module.Session.__init__ = custom_init
        return module


sys.meta_path.insert(0, CustomFinder())
