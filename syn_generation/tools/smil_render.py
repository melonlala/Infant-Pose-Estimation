"""pyrender (EGL, headless) replacement for the OpenDR renderer of render/image_generation.py.

All geometry is given in the *fitter's camera frame* (x right, y down, z forward, the
convention of smplifyx/camera.py and OpenDR). It is converted to OpenGL (y up, z back) here,
and the camera is an IntrinsicsCamera with the same K, so a vertex projected with
smil_common.project() lands exactly on its rendered pixel.
"""
import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np
import pyrender

from smil_common import ROOT

_GL = np.diag([1.0, -1.0, -1.0])


def load_template_uv(path=os.path.join(ROOT, 'render', 'template.obj')):
    """UVs of the SMIL template (render/template.obj, same faces as smil_web.pkl)."""
    vt, ft = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith('vt '):
                vt.append([float(x) for x in line.split()[1:3]])
            elif line.startswith('f '):
                ft.append([int(c.split('/')[1]) - 1 for c in line.split()[1:4]])
    return np.array(vt), np.array(ft)


def vertex_normals(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)


class Renderer:
    def __init__(self, width, height):
        self.W, self.H = width, height
        self.r = pyrender.OffscreenRenderer(width, height)

    def resize(self, width, height):
        if (width, height) != (self.W, self.H):
            self.r.delete()
            self.__init__(width, height)

    def render(self, verts_cam, faces, focal, center, texture=None, vt=None, ft=None,
               color=(0.85, 0.75, 0.65), lights=None, ambient=0.35):
        """verts_cam: (V,3) in camera frame. Returns rgb (H,W,3) uint8, mask (H,W) bool, depth (H,W)
        (camera-frame z, 0 = background)."""
        v = verts_cam @ _GL.T
        n = vertex_normals(verts_cam, faces) @ _GL.T
        if texture is not None:
            # un-weld: one vertex per face corner so UV seams are kept; smooth normals reused
            pos = v[faces.ravel()].astype(np.float32)
            nor = n[faces.ravel()].astype(np.float32)
            uv = vt[ft.ravel()].astype(np.float32)
            idx = np.arange(len(pos), dtype=np.uint32).reshape(-1, 3)
            tex = pyrender.Texture(source=texture, source_channels='RGB')
            mat = pyrender.MetallicRoughnessMaterial(baseColorTexture=tex, metallicFactor=0.0,
                                                     roughnessFactor=0.9, doubleSided=True)
            prim = pyrender.Primitive(positions=pos, normals=nor, texcoord_0=uv, indices=idx,
                                      material=mat)
        else:
            mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=(*color, 1.0), metallicFactor=0.0,
                                                     roughnessFactor=0.8, doubleSided=True)
            prim = pyrender.Primitive(positions=v.astype(np.float32), normals=n.astype(np.float32),
                                      indices=faces.astype(np.uint32), material=mat)
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[ambient] * 3)
        scene.add(pyrender.Mesh([prim]))
        cam = pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=center[0], cy=center[1],
                                        znear=0.01, zfar=100.0)
        scene.add(cam, pose=np.eye(4))
        if lights is None:
            lights = [dict(dir=(0.0, 0.0, 1.0), intensity=3.0, color=(1, 1, 1))]
        for L in lights:
            d = np.asarray(L['dir'], float) @ _GL.T          # direction the light travels (cam frame)
            d /= np.linalg.norm(d)
            z = -d                                            # pyrender lights shine along -z
            x = np.cross([0, 1, 0], z)
            if np.linalg.norm(x) < 1e-6:
                x = np.array([1.0, 0, 0])
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            pose = np.eye(4)
            pose[:3, :3] = np.stack([x, y, z], 1)
            scene.add(pyrender.DirectionalLight(color=L.get('color', (1, 1, 1)),
                                                intensity=L['intensity']), pose=pose)
        rgba, depth = self.r.render(scene, flags=pyrender.RenderFlags.RGBA)
        return rgba[..., :3], rgba[..., 3] > 0, depth


def render_labels(rend, verts_cam, faces, face_labels, focal, center):
    """Render an integer per-face label map (e.g. SMIL body part = dominant skinning joint).
    Returns (H,W) int array, -1 = background. Uses flat shading; labels are encoded in the red
    channel as 8*label+4, so MSAA-blended edge pixels can be rounded / filtered by the caller."""
    v = verts_cam @ _GL.T
    pos = v[faces.ravel()].astype(np.float32)
    lab = np.repeat(face_labels, 3)
    col = np.zeros((len(pos), 4), np.float32)
    col[:, 0] = (8 * lab + 4) / 255.0
    col[:, 3] = 1.0
    idx = np.arange(len(pos), dtype=np.uint32).reshape(-1, 3)
    mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=(1, 1, 1, 1), doubleSided=True)
    prim = pyrender.Primitive(positions=pos, color_0=col, indices=idx, material=mat)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1, 1, 1])
    scene.add(pyrender.Mesh([prim]))
    scene.add(pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=center[0], cy=center[1], znear=0.01, zfar=100.0),
              pose=np.eye(4))
    rgba, _ = rend.r.render(scene, flags=pyrender.RenderFlags.FLAT | pyrender.RenderFlags.RGBA)
    out = np.full(rgba.shape[:2], -1, int)
    m = rgba[..., 3] > 127
    out[m] = (rgba[..., 0][m].astype(int) - 4 + 4) // 8
    return out
