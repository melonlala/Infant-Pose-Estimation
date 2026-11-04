"""Load the original SMIL pickles (Python 2, chumpy objects) without chumpy.

`smil_web.pkl` and `smil_pose_prior.pkl` were written with Python 2 + chumpy.
chumpy does not install on python 3.13 / numpy 2, so this module unpickles them
with a *restricted* unpickler: only numpy / scipy.sparse array classes are
allowed, and the chumpy / Mahalanobis classes are replaced by plain stub objects
that just keep their pickled state. `to_array` then turns a stub into a numpy
array (a chumpy `Ch` keeps its value in `.x`; `Select` = `a.ravel()[idxs]`).
"""
import pickle

import numpy as np
import scipy.sparse


class _Stub:
    """Placeholder for chumpy.ch.Ch / chumpy.reordering.Select / Mahalanobis."""
    _kind = 'stub'

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2:  # (dict, slots)
            state = {**(state[0] or {}), **(state[1] or {})}
        self.__dict__.update(state if isinstance(state, dict) else {'state': state})


def _stub(kind):
    return type(kind, (_Stub,), {'_kind': kind})


_STUBS = {
    ('chumpy.ch', 'Ch'): _stub('Ch'),
    ('chumpy.reordering', 'Select'): _stub('Select'),
    ('__main__', 'Mahalanobis'): _stub('Mahalanobis'),
}
_ALLOWED = {
    ('numpy.core.multiarray', '_reconstruct'), ('numpy', 'ndarray'), ('numpy', 'dtype'),
    ('scipy.sparse.csc', 'csc_matrix'), ('__builtin__', 'set'), ('__builtin__', 'object'),
    ('copy_reg', '_reconstructor'), ('_codecs', 'encode'),
}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _STUBS:
            return _STUBS[(module, name)]
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f'blocked class {module}.{name}')
        if module == 'numpy.core.multiarray':
            return getattr(np._core.multiarray if hasattr(np, '_core') else np.core.multiarray, name)
        if module == 'numpy':
            return getattr(np, name)
        if module == 'scipy.sparse.csc':
            return scipy.sparse.csc_matrix
        if module == '__builtin__':
            return {'set': set, 'object': object}[name]
        if module == 'copy_reg':
            import copyreg
            return copyreg._reconstructor
        if module == '_codecs':
            import codecs
            return codecs.encode
        raise pickle.UnpicklingError(f'blocked class {module}.{name}')


def load(path):
    with open(path, 'rb') as f:
        return SafeUnpickler(f, encoding='latin1').load()


def to_array(obj):
    """chumpy stub / sparse / array -> numpy array (sparse stays sparse)."""
    if isinstance(obj, _Stub):
        if obj._kind == 'Ch':
            return np.asarray(obj.x)
        if obj._kind == 'Select':
            a = to_array(obj.a)
            out = a.ravel()[np.asarray(obj.idxs)]
            shape = getattr(obj, 'preferred_shape', None)
            return out.reshape(shape) if shape is not None else out
        raise TypeError(f'cannot convert stub {obj._kind}: {list(obj.__dict__)}')
    return obj
