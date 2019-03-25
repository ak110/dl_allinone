def test_run():
    from numpy.distutils.system_info import get_info

    assert "/opt/intel/mkl/lib/intel64" in get_info("blas_mkl")["library_dirs"]
