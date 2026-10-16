"""
src/fit/fit_worker.py
=====================
Long-running body-fit worker. Run it with the Python 3.7 + TensorFlow 1.15 ("hmr")
environment. It is normally started automatically by src/height/pipeline.py.

It loads DeepLab + HMR once, then waits on a local socket for frames:
    request : {"cmd": "measure", "height_cm": <float>}  + JPEG bytes (BGR frame)
    reply   : {"ok": true, "measurements": {name: value, ...}, "elapsed_s": <float>}
              {"ok": false, "error": "<message>"}
The worker exits when the client disconnects or sends {"cmd": "quit"}.
"""
import argparse
import json
import os
import socket
import struct
import sys
import tarfile
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)            # data/ and ./src/... paths are relative to src/fit
sys.path.insert(0, HERE)

# HMR was fed BGR by the original inference.py. Set True to feed RGB instead.
HMR_INPUT_RGB = True
PERSON_LABEL = 15
MIN_PERSON_FRACTION = 0.02


# ---------- tiny framing protocol (identical in src/height/fit_client.py) ----------

def _send(sock, header, blob=b""):
    j = json.dumps(header).encode("utf-8")
    sock.sendall(struct.pack(">II", len(j), len(blob)) + j + blob)


def _recvn(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed")
        buf.extend(chunk)
    return bytes(buf)


def _recv(sock):
    jl, bl = struct.unpack(">II", _recvn(sock, 8))
    header = json.loads(_recvn(sock, jl).decode("utf-8"))
    blob = _recvn(sock, bl) if bl else b""
    return header, blob


# ---------- engines ----------

class DummyEngine(object):
    """Used for testing the plumbing without TensorFlow."""

    def measure(self, jpg_bytes, height_cm):
        time.sleep(1.0)
        return {"ok": True, "elapsed_s": 1.0,
                "measurements": {"Height": height_cm, "Chest": height_cm * 0.55,
                                 "Waist": height_cm * 0.45}}


class FitEngine(object):
    DEEPLAB_URL = 'http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz'
    INPUT_SIZE = 513

    def __init__(self):
        import cv2
        from PIL import Image
        import tensorflow as tf
        self.cv2 = cv2
        self.Image = Image
        self.tf = tf
        self.resample = getattr(Image, 'LANCZOS', None) or getattr(Image, 'ANTIALIAS')

        print("[fit_worker] importing HMR...", flush=True)
        import src.config as hmr_config  # raises a clear error if models/ is missing
        from absl import flags
        flags.FLAGS(sys.argv[:1])          # parse (empty) command line so flags are readable
        cfg = flags.FLAGS
        if cfg.load_path is None:          # same settings as the original HMR demo
            cfg.load_path = hmr_config.PRETRAINED_MODEL
        cfg.batch_size = 1
        from src.RunModel import RunModel
        from src.util import image as img_util
        import extract_measurements
        self.img_util = img_util
        self.em = extract_measurements

        print("[fit_worker] loading DeepLab...", flush=True)
        self._load_deeplab()

        print("[fit_worker] loading HMR model...", flush=True)
        self.sess = tf.Session()
        self.model = RunModel(sess=self.sess)

    def _load_deeplab(self):
        import urllib.request
        tf = self.tf
        d = os.path.join(HERE, 'deeplab_model')
        if not os.path.exists(d):
            os.makedirs(d)
        tar_path = os.path.join(d, os.path.basename(self.DEEPLAB_URL))
        if not os.path.exists(tar_path):
            print("[fit_worker] downloading DeepLab to %s ..." % tar_path, flush=True)
            urllib.request.urlretrieve(self.DEEPLAB_URL, tar_path)

        graph_def = None
        tar_file = tarfile.open(tar_path)
        for info in tar_file.getmembers():
            if 'frozen_inference_graph' in os.path.basename(info.name):
                graph_def = tf.GraphDef.FromString(tar_file.extractfile(info).read())
                break
        tar_file.close()
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in DeepLab tar archive.')

        self.dl_graph = tf.Graph()
        with self.dl_graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.dl_sess = tf.Session(graph=self.dl_graph)

    def _person_mask(self, bgr):
        cv2, Image = self.cv2, self.Image
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        w, h = pil.size
        ratio = 1.0 * self.INPUT_SIZE / max(w, h)
        resized = pil.resize((int(ratio * w), int(ratio * h)), self.resample)
        seg = self.dl_sess.run('SemanticPredictions:0',
                               feed_dict={'ImageTensor:0': [np.asarray(resized)]})[0]
        seg = cv2.resize(seg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return ((seg == PERSON_LABEL) * 255).astype(np.uint8)

    def _preprocess(self, img):
        if img.shape[2] == 4:
            img = img[:, :, :3]
        longest = np.max(img.shape[:2])
        scale = (224.0 / longest) if longest != 224 else 1.0
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)[::-1]
        crop, _ = self.img_util.scale_and_crop(img, scale, center, 224)
        return 2 * ((crop / 255.) - 0.5)

    def measure(self, jpg_bytes, height_cm):
        cv2 = self.cv2
        t0 = time.time()
        bgr = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return {"ok": False, "error": "Could not decode frame"}

        mask = self._person_mask(bgr)
        if mask.mean() / 255.0 < MIN_PERSON_FRACTION:
            return {"ok": False, "error": "No person found in the frame"}

        res = cv2.bitwise_and(bgr, bgr, mask=mask)
        bg_removed = res + (255 - cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        img = np.ascontiguousarray(bg_removed[:, :, ::-1]) if HMR_INPUT_RGB else bg_removed

        crop = self._preprocess(img)
        joints, verts, cams, joints3d, theta = self.model.predict(
            np.expand_dims(crop, 0), get_theta=True)

        values = self.em.compute_measurements(height_cm, verts[0])
        return {"ok": True, "measurements": values, "elapsed_s": round(time.time() - t0, 1)}


def serve(engine, port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    print("[fit_worker] READY on port %d" % port, flush=True)
    conn, _ = srv.accept()
    try:
        while True:
            try:
                header, blob = _recv(conn)
            except (ConnectionError, OSError):
                break
            cmd = header.get("cmd")
            if cmd == "quit":
                break
            if cmd == "measure":
                try:
                    reply = engine.measure(blob, float(header["height_cm"]))
                except Exception as e:
                    traceback.print_exc()
                    reply = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
                _send(conn, reply)
            else:
                _send(conn, {"ok": False, "error": "unknown command"})
    finally:
        conn.close()
        srv.close()
        print("[fit_worker] exiting", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--dummy", action="store_true", help="skip TensorFlow (plumbing test)")
    a = p.parse_args()
    sys.argv = sys.argv[:1]   # keep absl flags from seeing our arguments
    engine = DummyEngine() if a.dummy else FitEngine()
    serve(engine, a.port)


if __name__ == "__main__":
    main()