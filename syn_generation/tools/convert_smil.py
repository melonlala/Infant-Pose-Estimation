"""One-off conversion of the SMIL pickles to plain numpy files (no chumpy needed).

    cd syn_generation
    ../.venv-smil/bin/python tools/convert_smil.py

Reads   models/smil_web.pkl, priors/smil_pose_prior.pkl (original Python-2/chumpy pickles)
Writes  models/smil_web_np.npz        v_template, shapedirs, posedirs, J_regressor (dense), weights,
                                      kintree_table, f, J   -- used by smplifyx/smil_model.py and the renderer
        render/smil_web_np.npz        copy for the renderer
        priors/smil_pose_prior_np.npz mean (69,), prec (69,69)  (Mahalanobis: |(pose[3:]-mean) @ prec|^2)
        priors/gmm_01.pkl             the SMIL Gaussian pose prior written as a 1-component GMM in the
                                      SMPLify format {means, covars, weights}, so `body_prior_type: gmm`,
                                      `num_gaussians: 1` works (the SMPL gmm_08.pkl the authors used is not
                                      distributed with SMIL).
"""
import os.path as osp
import pickle
import shutil
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import smil_pickle  # noqa: E402

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))


def main():
    d = smil_pickle.load(osp.join(ROOT, 'models', 'smil_web.pkl'))
    out = {}
    for k in ['v_template', 'shapedirs', 'posedirs', 'J_regressor', 'weights',
              'kintree_table', 'f', 'J']:
        v = smil_pickle.to_array(d[k])
        if hasattr(v, 'toarray'):
            v = v.toarray()
        out[k] = np.asarray(v)
    out['f'] = out['f'].astype(np.int64)
    out['kintree_table'] = out['kintree_table'].astype(np.int64)
    for k, v in out.items():
        print(f'{k:14s} {v.dtype} {v.shape}')
    # sanity: J_regressor applied to the template should give the stored rest joints J
    err = np.abs(out['J_regressor'] @ out['v_template'] - out['J']).max()
    print('max |J_regressor @ v_template - J| =', err)
    dst = osp.join(ROOT, 'models', 'smil_web_np.npz')
    np.savez(dst, **out)
    shutil.copy(dst, osp.join(ROOT, 'render', 'smil_web_np.npz'))

    p = smil_pickle.load(osp.join(ROOT, 'priors', 'smil_pose_prior.pkl'))
    mean = np.asarray(p.mean, dtype=np.float64)
    prec = smil_pickle.to_array(p.prec).astype(np.float64)
    print('prior prefix', p.prefix, 'mean', mean.shape, 'prec', prec.shape)
    np.savez(osp.join(ROOT, 'priors', 'smil_pose_prior_np.npz'), mean=mean, prec=prec, prefix=p.prefix)
    # Mahalanobis distance = (x-m) prec prec^T (x-m)^T  ->  precision matrix P = prec prec^T
    P = prec @ prec.T
    cov = np.linalg.pinv(P)
    cov = 0.5 * (cov + cov.T) + 1e-8 * np.eye(len(mean))  # some dims (hands/feet) have ~0 variance
    gmm = {'means': mean[None], 'covars': cov[None], 'weights': np.array([1.0])}
    with open(osp.join(ROOT, 'priors', 'gmm_01.pkl'), 'wb') as f:
        pickle.dump(gmm, f, protocol=4)
    print('eig(P) min/max', np.linalg.eigvalsh(P).min(), np.linalg.eigvalsh(P).max())


if __name__ == '__main__':
    main()
