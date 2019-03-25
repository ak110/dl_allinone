def test_run():
    from scipy.__config__ import get_info

    assert "/opt/intel/mkl/lib/intel64" in get_info("blas_mkl")["library_dirs"]
