"""GPU availability diagnostics; the GA fitness path itself remains CPU-based."""
import importlib.util, platform

def cuda_info():
    info={"has_cupy":False, "has_numba":False, "cupy_version":None, "numba_version":None, "platform":platform.platform()}
    if importlib.util.find_spec("cupy") is not None:
        try:
            import cupy
            info["has_cupy"]=True
            info["cupy_version"]=cupy.__version__
            # quick alloc test
            cp=cupy
            a=cp.zeros((6,6)); b=cp.eye(6); c=cp.matmul(a,b); cp.cuda.Stream.null.synchronize()
        except Exception as e:
            info["cupy_error"]=str(e)
            info["has_cupy"]=False
    if importlib.util.find_spec("numba") is not None:
        try:
            import numba
            info["has_numba"]=True
            info["numba_version"]=numba.__version__
            info["numba_cuda_available"]=bool(numba.cuda.is_available()) if hasattr(numba.cuda, "is_available") else False
        except Exception as e:
            info["numba_error"]=str(e)
    return info

def get_xp():
    """Optional array-module helper; current GA fitness code does not call it."""
    try:
        import cupy as cp
        # test
        cp.zeros((2,2))
        return cp
    except Exception:
        import numpy as np
        return np
