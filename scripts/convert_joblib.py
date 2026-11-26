"""Re-save .joblib files with the scikit-learn installed in the current environment.

Pickles written by an older scikit-learn can fail to load in a newer one. The models'
`hist_gbdt.joblib` classifiers were saved with scikit-learn 1.3.2, which cannot be installed on
Python 3.13; newer versions no longer ship the Cython helpers that rebuild the classifier's loss
object (`__pyx_unpickle_CyHalfBinomialLoss` errors). This script provides those helpers while
loading, checks that the re-saved file gives the same predictions, and writes it back in the
format of the installed version. The original is kept next to it as
`<name>.sklearn-<old version>.joblib`, so the conversion can be undone by renaming it back.

Run it from the repository root, in the environment that will use the models:

    python scripts/convert_joblib.py                   # every .joblib under BASE_MODEL_PATH
    python scripts/convert_joblib.py <dir or file> ... # only these
    python scripts/convert_joblib.py --dry-run         # report what would be converted, write nothing
"""
import argparse
import io
import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import sklearn
import sklearn._loss._loss as sk_loss
from sklearn.exceptions import InconsistentVersionWarning

BACKUP_TAG = ".sklearn-"


@contextmanager
def legacy_loss_unpicklers():
    """Let old pickles of the Cython loss classes load on newer scikit-learn.

    Old versions pickled a loss as `__pyx_unpickle_<Class>(cls, checksum, state)` where `state`
    is exactly the constructor arguments (`()` for the binomial loss, `(delta,)` for Huber, ...),
    and newer versions pickle the same object as `<Class>(*state)`. The helpers are removed again
    on exit, so a re-saved file is checked without them.
    """
    replaced = {}
    for name in dir(sk_loss):
        cls = getattr(sk_loss, name)
        if not (name.startswith("Cy") and isinstance(cls, type)):
            continue
        key = f"__pyx_unpickle_{name}"
        original = getattr(sk_loss, key, None)

        def unpickle(cls, checksum, state, original=original):
            if original is not None:
                try:
                    return original(cls, checksum, state)
                except Exception:      # checksum of another scikit-learn version
                    pass
            return cls(*state)

        replaced[key] = original
        setattr(sk_loss, key, unpickle)
    try:
        yield
    finally:
        for key, original in replaced.items():
            if original is None:
                delattr(sk_loss, key)
            else:
                setattr(sk_loss, key, original)


def load_with_version(path):
    """Load a pickle and return (object, scikit-learn version that saved it)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        try:
            obj = joblib.load(path)
        except Exception:
            with legacy_loss_unpicklers():
                obj = joblib.load(path)
    saved = [w.message.original_sklearn_version for w in caught if issubclass(w.category, InconsistentVersionWarning)]
    return obj, (saved[0] if saved else sklearn.__version__)


def check_predictions(old, new):
    """Max difference between the two objects' predictions on inputs spread over each feature's range."""
    if not hasattr(old, "_bin_mapper"):          # only HistGradientBoosting* know their feature ranges
        return None
    thresholds = old._bin_mapper.bin_thresholds_
    low = np.array([t.min() for t in thresholds])
    high = np.array([t.max() for t in thresholds])
    span = high - low
    X = np.random.default_rng(0).uniform(low - 0.1 * span, high + 0.1 * span, size=(20000, len(thresholds)))
    predict = "predict_proba" if hasattr(old, "predict_proba") else "predict"
    return float(np.max(np.abs(getattr(old, predict)(X) - getattr(new, predict)(X))))


def convert(path, dry_run):
    obj, saved_version = load_with_version(path)
    if saved_version == sklearn.__version__:
        print(f"  up to date (scikit-learn {saved_version}): {path}")
        return
    backup = path.with_name(f"{path.stem}{BACKUP_TAG}{saved_version}{path.suffix}")
    if dry_run:
        print(f"  would convert scikit-learn {saved_version} -> {sklearn.__version__}: {path}")
        return
    if backup.exists():
        print(f"  skipped, backup {backup.name} already exists: {path}")
        return

    # Older HistGradientBoostingClassifier pickles lack `_preprocessor` on newer scikit-learn.
    if obj.__class__.__name__.startswith("HistGradientBoosting") and not hasattr(obj, "_preprocessor"):
        obj._preprocessor = None

    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    buffer.seek(0)
    reloaded = joblib.load(buffer)               # without the legacy helpers
    diff = check_predictions(obj, reloaded)
    if diff is not None and diff > 0:
        raise RuntimeError(f"re-saved {path} predicts differently (max diff {diff:.3g}); nothing written")

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(buffer.getvalue())
    os.replace(path, backup)
    os.replace(tmp, path)
    check = "" if diff is None else ", predictions identical"
    print(f"  converted scikit-learn {saved_version} -> {sklearn.__version__}{check}: {path} (original: {backup.name})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="*", help=".joblib files or directories searched recursively (default: BASE_MODEL_PATH)")
    parser.add_argument("--dry-run", action="store_true", help="report what would be converted, write nothing")
    args = parser.parse_args()

    paths = args.paths
    if not paths:
        sys.path.append("scripts/")
        from local_paths import BASE_MODEL_PATH
        paths = [BASE_MODEL_PATH]

    files = []
    for p in map(Path, paths):
        found = sorted(p.rglob("*.joblib")) if p.is_dir() else [p]
        files += [f for f in found if BACKUP_TAG not in f.name]

    print(f"scikit-learn {sklearn.__version__}, {len(files)} .joblib file(s)")
    for f in files:
        convert(f, args.dry_run)


if __name__ == "__main__":
    main()
