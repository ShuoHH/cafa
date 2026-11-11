"""Raw-only shared-bank policies and a train-only boundary20 comparison.

Compact20/base10 models are supplied by the unchanged authenticated old loader.
Boundary20 is fitted once on the SAME ten old sources, regions and certified
retained-eight labels. New monitoring/medical sources never enter fitting,
imputation, weights or model selection. No codecs or measured test costs enter
this module. Shared-bank preprocessing must be charged by its caller.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import importlib
import json
from numbers import Integral
from pathlib import Path
import time
from codec_pool import RETAINED_ACTIONS, RETAINED_STREAM_ACTIONS


VERSION = "shared_bank_frozen_and_old_train_boundary20_policy_v1"
KEY = ("dataset", "series", "sha256", "start", "length")
ANCHOR_SERIES = "202005_SPlite2"
ACTIONS = RETAINED_ACTIONS
ORDER = ACTIONS + tuple("alp+tail:" + a for a in RETAINED_STREAM_ACTIONS)


def _bank_api():
    return importlib.import_module("shared_compression_features")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _identity(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 3
            or any(not isinstance(s, str) or not s for s in value)
            or len(value[2]) != 64 or any(c not in "0123456789abcdef" for c in value[2])):
        raise ValueError("require dataset/series/lowercase raw SHA identity")
    return tuple(value)


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= 102400:
        raise ValueError(name + " must be an integer within the unchanged native range")
    return int(value)


def feature_columns(variant):
    bank = _bank_api()
    if variant in ("compact20", "original20", "frozen20"):
        return list(bank.COMPACT_COLUMNS)
    if variant == "boundary20":
        return list(bank.BOUNDARY20_COLUMNS)
    if variant == "base10":
        return list(bank.BASE_COLUMNS)
    raise ValueError("only compact20, boundary20 and base10 are predeclared")


def _feasible(length):
    length = _integer(length, "length")
    if length % 1024 == 0:
        return ACTIONS
    return tuple(a for a in ACTIONS if a != "alp") + (ORDER[len(ACTIONS):] if length > 1024 else ())


def _check(protected):
    for path, digest in protected.items():
        if _sha(path) != digest:
            raise ValueError("authenticated source/model changed: " + str(path))


def _metadata_sources(metadata, anchor_series):
    heldout = _identity(metadata.get("held_out_identity"))
    sources = [_identity(s) for s in metadata.get("training_sources", [])]
    if (heldout[1] != anchor_series or len(sources) != 10 or len(set(sources)) != 10
            or len({s[1] for s in sources}) != 10 or len({s[2] for s in sources}) != 10
            or any(s[1] == heldout[1] or s[2] == heldout[2] for s in sources)):
        raise ValueError("require the unchanged complete-source/SHA ten-source anchor fold")
    if (metadata.get("arm") != "class_balanced" or metadata.get("top_k") != 2
            or any(metadata.get(k) is not False for k in
                   ("test_costs_used", "test_labels_used", "test_support_status_used"))):
        raise ValueError("model must certify class-balanced training without held-out prices/labels/support")
    return heldout, sources


def _validate_model(model, metadata, columns):
    import numpy as np
    from rf_profiles import RF_REPRO_PARAMS
    if (metadata.get("feature_columns") != list(columns)
            or metadata.get("feature_count") != len(columns)
            or metadata.get("rf_parameters") != RF_REPRO_PARAMS
            or model.n_features_in_ != len(columns)):
        raise ValueError("model dimensions/column order/RF profile differ")
    classes = tuple(map(str, model.classes_))
    if (not classes or len(classes) != len(set(classes))
            or list(classes) != metadata.get("classes") or any(a not in ORDER for a in classes)
            or not callable(getattr(model, "predict_proba", None))):
        raise ValueError("model classes differ from the fixed retained-pool labels")
    forest, imputer = model.named_steps["forest"], model.named_steps["imputer"]
    if any(forest.get_params().get(k) != v for k, v in RF_REPRO_PARAMS.items()):
        raise ValueError("saved forest differs from the unchanged RF settings")
    if (metadata.get("training_only_median_imputation") is not True
            or imputer.strategy != "median" or not imputer.keep_empty_features
            or not np.array_equal(np.asarray(imputer.statistics_),
                np.asarray(metadata.get("imputer_medians")), equal_nan=True)):
        raise ValueError("saved training-only median imputer differs")
    if "model_path" in metadata or "model_sha256" in metadata:
        if _sha(metadata["model_path"]) != metadata["model_sha256"]:
            raise ValueError("saved model artifact SHA differs from authenticated metadata")
    return classes


class FactoryPolicy:
    """One source's fresh recommendation cache over a caller-owned shared bank.

    The caller authenticates and loads model artifacts before construction.
    The actual old held-out identity is retained; external raw sources are not
    relabelled as new held-out folds. A bank is never constructed silently here.
    """
    def __init__(self, job, model, metadata, bank, columns=None, top_k=2, *,
                 batch_size=16, on_freeze=None):
        import numpy as np
        if type(top_k) is not int or top_k not in (1, 2):
            raise ValueError("predeclare Top-1 or Top-2")
        if type(batch_size) is not int or not 1 <= batch_size <= 32:
            raise ValueError("batch_size must be in 1..32")
        self.identity = _identity(tuple(job[k] for k in KEY[:3]))
        self.raw, self.values = Path(job["input"]).resolve(), _integer(job["values"], "values")
        self.columns = list(metadata.get("feature_columns", []) if columns is None else columns)
        kind = metadata.get("feature_set")
        if kind not in ("compact20", "boundary20", "base10") or self.columns != feature_columns(kind):
            raise ValueError("requested columns differ from the predeclared trained feature variant")
        heldout, sources = _metadata_sources(metadata, metadata["held_out_identity"][1])
        if any(s[1] == self.identity[1] or s[2] == self.identity[2] for s in sources):
            raise ValueError("requested raw source leaks into model training")
        self.external = self.identity != heldout
        if self.external and (self.identity[1] == heldout[1] or self.identity[2] == heldout[2]):
            raise ValueError("external source aliases the old held-out source")
        if self.external and any(s[0] == self.identity[0] for s in sources + [heldout]):
            raise ValueError("external dataset is not disjoint from the old model sources")
        if (Path(bank.raw).resolve() != self.raw or tuple(bank.identity) != self.identity
                or bank.n != self.values or bank.source_sha256 != self.identity[2]):
            raise ValueError("shared bank source/path/count/SHA differs from the true raw job")
        if (self.raw.stat().st_size != 8 * self.values
                or metadata.get("feature_bank_version", _bank_api().VERSION) != _bank_api().VERSION):
            raise ValueError("raw byte count or trained shared-bank version differs")
        self.classes = _validate_model(model, metadata, self.columns)
        if on_freeze is not None and not callable(on_freeze):
            raise ValueError("on_freeze must be callable")
        self.model, self.bank, self.metadata = model, bank, copy.deepcopy(metadata)
        self.original_metadata = copy.deepcopy(metadata)
        self.original_held_out_identity, self.original_training_sources = heldout, sources
        self.top_k, self.batch_size, self._np = top_k, batch_size, np
        self._on_freeze, self._frozen, self._predictions = on_freeze, {}, []
        self.counters = dict(raw_checks=0, raw_check_seconds=0.0, requests=0, cache_hits=0,
            feature_batches=0, feature_regions=0, feature_values=0, feature_seconds=0.0,
            inference_batches=0, inference_regions=0, inference_seconds=0.0, frozen_regions=0,
            shared_feature_bank_used=True, bank_preprocessing_cost_must_be_charged_by_caller=True)
        self._check_raw()

    @property
    def predictions(self):
        return copy.deepcopy(self._predictions)

    def _check_raw(self):
        began = time.perf_counter()
        self.bank.assert_unchanged()
        self.counters["raw_checks"] += 1
        self.counters["raw_check_seconds"] += time.perf_counter() - began

    def _region(self, region):
        start, length = (region["start"], region["length"]) if isinstance(region, dict) else region
        start, length = _integer(start, "start", 0), _integer(length, "length")
        if start + length > self.values:
            raise ValueError("feature region exceeds its original source")
        return start, length

    def actions_for(self, region):
        return self.predict_many([region])[0]

    def predict_many(self, regions):
        parsed = [self._region(r) for r in regions]
        if not parsed:
            return []
        self._check_raw()
        self.counters["requests"] += len(parsed)
        pending = list(dict.fromkeys(r for r in parsed if r not in self._frozen))
        self.counters["cache_hits"] += len(parsed) - len(pending)
        for offset in range(0, len(pending), self.batch_size):
            self._predict_batch(pending[offset:offset + self.batch_size])
        return [self._frozen[r] for r in parsed]

    def _predict_batch(self, regions):
        np = self._np
        began = time.perf_counter()
        frame = self.bank.frame(regions, self.columns)
        self.counters["feature_seconds"] += time.perf_counter() - began
        self.counters["feature_batches"] += 1
        self.counters["feature_regions"] += len(regions)
        self.counters["feature_values"] += sum(n for _, n in regions)
        if (list(frame.columns) != list(KEY) + self.columns
                or list(frame[list(KEY)].itertuples(index=False, name=None))
                   != [self.identity + r for r in regions]
                or frame.attrs.get("source_sha256") != self.identity[2]
                or np.isinf(frame[self.columns].to_numpy(dtype=float)).any()):
            raise ValueError("shared feature frame identity/order/schema differs or contains infinity")
        began = time.perf_counter()
        probability = np.asarray(self.model.predict_proba(frame[self.columns]), dtype=float)
        self.counters["inference_seconds"] += time.perf_counter() - began
        self.counters["inference_batches"] += 1
        self.counters["inference_regions"] += len(regions)
        if (probability.shape != (len(regions), len(self.classes))
                or not np.isfinite(probability).all() or (probability < 0).any()
                or not np.allclose(probability.sum(axis=1), 1, rtol=1e-10, atol=1e-12)):
            raise ValueError("saved model produced invalid class probabilities")
        for i, (start, length) in enumerate(regions):
            values = {a: float(probability[i, j]) for j, a in enumerate(self.classes)}
            legal = set(_feasible(length))
            ranked = sorted(((a, p) for a, p in values.items() if a in legal),
                            key=lambda item: (-item[1], ORDER.index(item[0])))
            record = dict(zip(KEY[:3], self.identity), start=start, length=length,
                feature_set=self.metadata["feature_set"], feature_count=len(self.columns),
                feature_columns=self.columns, top_k=self.top_k, policy_version=VERSION,
                ranked_actions=[a for a, _ in ranked[:self.top_k]],
                probabilities=[p for _, p in ranked[:self.top_k]],
                all_ranked_actions=[a for a, _ in ranked],
                all_ranked_probabilities=[p for _, p in ranked], class_probabilities=values,
                recommended_count=min(self.top_k, len(ranked)),
                external_model_reuse=self.external, new_held_out_fold=False,
                original_held_out_identity=list(self.original_held_out_identity),
                original_training_sources=[list(s) for s in self.original_training_sources],
                model_refit_at_prediction=False, test_costs_used=False,
                test_labels_used=False, test_support_status_used=False,
                geometry_feasibility_only=True, frozen_before_native=True,
                shared_raw_feature_bank=True, hidden_fallback_used=False)
            if self._on_freeze is not None:
                self._on_freeze(copy.deepcopy(record))
            self._predictions.append(record)
            self._frozen[start, length] = tuple(record["ranked_actions"])
            self.counters["frozen_regions"] += 1


SharedFeaturePolicy = FactoryPolicy


def make_policy(job, model, metadata, bank, variant="compact20", top_k=2, **kwargs):
    expected = "compact20" if variant in ("original20", "frozen20") else variant
    if metadata.get("feature_set") != expected:
        raise ValueError("requested variant differs from the saved model metadata")
    return FactoryPolicy(job, model, metadata, bank, feature_columns(expected), top_k, **kwargs)


def _certified_path(prepared, filename):
    protected = {str(Path(p).resolve()): h for p, h in prepared["protected"].items()}
    preferred = Path(prepared.get("manifest", {}).get("original", ".")) / filename
    path = preferred.resolve()
    if str(path) not in protected:
        candidates = [Path(p) for p in protected if Path(p).name == filename]
        if len(candidates) != 1:
            raise ValueError("require one authenticated original " + filename)
        path = candidates[0]
    if _sha(path) != protected[str(path)]:
        raise ValueError("original training certificate changed: " + filename)
    return path, protected[str(path)]


def _training_context(prepared, anchor_series):
    import pandas as pd
    if anchor_series != ANCHOR_SERIES:
        raise ValueError("the external comparison uses the predeclared May anchor only")
    meta = prepared["metadata"][anchor_series, "frozen20"]
    heldout, sources = _metadata_sources(meta, anchor_series)
    if meta.get("feature_columns") != feature_columns("compact20"):
        raise ValueError("reference compact20 column schema differs")
    source_csv, csv_sha = _certified_path(prepared, "retained8_features_and_labels.csv")
    plan_path, plan_sha = _certified_path(prepared, "frozen_source_plan.json")
    certified = pd.read_csv(source_csv, float_precision="round_trip", dtype={k: str for k in KEY[:3]})
    frame = prepared["frame"]
    def labels(rows):
        found = {}
        for r in rows.to_dict("records"):
            identity = _identity(tuple(r[k] for k in KEY[:3]))
            start, length = _integer(r["start"], "start", 0), _integer(r["length"], "length")
            key = identity + (start, length)
            label = r["target_action"]
            if key in found or label not in _feasible(length):
                raise ValueError("original training region labels are duplicated or geometrically infeasible")
            found[key] = label
        return found
    if (labels(frame) != labels(certified)
            or list(frame[list(KEY)].itertuples(index=False, name=None))
               != list(certified[list(KEY)].itertuples(index=False, name=None))):
        raise ValueError("prepared training-region labels differ from certified old retained-pool labels")
    all_sources = {_identity(tuple(r[k] for k in KEY[:3])) for r in frame.to_dict("records")}
    if all_sources != set(sources + [heldout]):
        raise ValueError("original feature/label registry is not exactly the old eleven sources")
    train = frame[(frame.series != heldout[1]) & (frame.sha256 != heldout[2])].copy()
    if len(train) != meta["training_rows"]:
        raise ValueError("old training-region union size differs")
    with plan_path.open(encoding="utf-8") as stream:
        plan = json.load(stream)
    jobs = {}
    for entry in plan["sources"]:
        job = entry["job"]
        identity = _identity(tuple(job[k] for k in KEY[:3]))
        if identity in jobs:
            raise ValueError("duplicate original training source job")
        jobs[identity] = job
    if set(jobs) != all_sources:
        raise ValueError("old raw source plan differs from certified label sources")
    bindings = {str(source_csv): csv_sha, str(plan_path): plan_sha}
    for identity in sources:
        job = jobs[identity]
        raw = Path(job["input"]).resolve()
        if raw.stat().st_size != 8 * _integer(job["values"], "values") or _sha(raw) != identity[2]:
            raise ValueError("old training raw source count/SHA differs")
        bindings[str(raw)] = identity[2]
    ordered = [list(tuple(r[k] for k in KEY) + (r["target_action"],)) for r in train.to_dict("records")]
    registry_sha = hashlib.sha256(json.dumps(ordered, separators=(",", ":")).encode()).hexdigest()
    return meta, heldout, sources, train, jobs, bindings, registry_sha


def train_boundary_variant(prepared, anchor_series, out, *, bank_factory=None):
    """Fit boundary10+base10 ONLY on authenticated original ten-source labels."""
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from rf_profiles import RF_REPRO_PARAMS
    reference, heldout, sources, train, jobs, bindings, registry_sha = _training_context(prepared, anchor_series)
    if reference.get("rf_parameters") != RF_REPRO_PARAMS:
        raise ValueError("reference RF settings differ from the frozen reproducibility profile")
    columns = feature_columns("boundary20")
    out = Path(out).resolve()
    if out.exists() or any(out == Path(p) or out in Path(p).parents or Path(p) in out.parents for p in bindings):
        raise ValueError("fresh nonoverlapping boundary20 model output required")
    factory = bank_factory or _bank_api().FeatureBank
    feature_rows = {}
    began = time.perf_counter()
    for identity in sources:
        source_rows = train[(train.series == identity[1]) & (train.sha256 == identity[2])]
        regions = [(int(r.start), int(r.length)) for r in source_rows.itertuples()]
        bank = factory(jobs[identity]["input"], identity)
        frame = bank.frame(regions, columns)
        if (list(frame.columns) != list(KEY) + columns or frame.attrs.get("source_sha256") != identity[2]):
            raise ValueError("training bank returned a foreign source/schema")
        for row in frame.to_dict("records"):
            key = tuple(row[k] for k in KEY)
            if key in feature_rows or key[:3] != identity:
                raise ValueError("training bank duplicated or relabelled a source region")
            feature_rows[key] = row
        bank.assert_unchanged()
    ordered = []
    for row in train.to_dict("records"):
        key = tuple(row[k] for k in KEY)
        if key not in feature_rows:
            raise ValueError("shared training bank omitted an original labelled region")
        ordered.append(dict(feature_rows[key], target_action=row["target_action"]))
    if len(feature_rows) != len(ordered):
        raise ValueError("shared training bank added uncertified regions")
    training = pd.DataFrame(ordered, columns=list(KEY) + columns + ["target_action"])
    if np.isinf(training[columns].to_numpy(dtype=float)).any():
        raise ValueError("training features contain infinity; no test-derived repair/imputation")
    counts, n = Counter(training.target_action), len(training)
    weights = np.asarray([n / (len(counts) * counts[a]) for a in training.target_action], dtype=np.float64)
    weights /= weights.mean()
    weight_sha = hashlib.sha256(weights.astype("<f8").tobytes()).hexdigest()
    old_weight_sha = reference.get("training_weights_sha256", reference.get("sample_weights_sha256"))
    if old_weight_sha is not None and old_weight_sha != weight_sha:
        raise ValueError("new training weights differ from the old class-balanced row weights")
    model = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                      ("forest", RandomForestClassifier(**RF_REPRO_PARAMS))])
    model.fit(training[columns], training.target_action, forest__sample_weight=weights)
    metadata = dict(version=VERSION, arm="class_balanced", feature_set="boundary20",
        feature_columns=columns, feature_count=20, top_k=2, training_rows=n,
        training_sources=[list(s) for s in sources], held_out_identity=list(heldout),
        classes=list(map(str, model.classes_)), training_class_counts=dict(counts),
        rf_parameters=dict(RF_REPRO_PARAMS), library_versions=copy.deepcopy(prepared.get("versions", {})),
        imputer_medians=[float(v) for v in model.named_steps["imputer"].statistics_],
        training_only_median_imputation=True, test_costs_used=False, test_labels_used=False,
        test_support_status_used=False, new_monitor_medical_training_data_used=False,
        unchanged_original_region_union=True, new_training_regions=False,
        retained_actions=list(ACTIONS), sample_weights_sha256=weight_sha,
        original_class_balanced_weights_unchanged=True,
        training_regions_sha256=registry_sha, training_source_sha256=bindings,
        feature_bank_version=_bank_api().VERSION, training_wall_seconds=time.perf_counter()-began,
        preprocessing_and_fit_included_in_training_time=True, test_timing_excludes_training=True,
        feature_definition_scope="original base10 plus all original boundary-state10; no feature selection")
    _check(bindings)
    out.mkdir(parents=True)
    model_path = out / "boundary20.joblib"
    joblib.dump(model, model_path, compress=3)
    metadata.update(model_path=str(model_path), model_sha256=_sha(model_path))
    _validate_model(model, metadata, columns)
    _write(out / "metadata.json", metadata)
    _write(out / "training_manifest.json", dict(version=VERSION,
        model_sha256=metadata["model_sha256"], metadata_sha256=_sha(out / "metadata.json"),
        training_regions_sha256=registry_sha, protected_training_sha256=bindings,
        new_monitor_medical_training_data_used=False, held_out_identity=list(heldout)))
    return model, metadata


def load_boundary_variant(prepared, folder):
    """Authenticate the new old-data-only B model before deserializing it."""
    import joblib
    folder = Path(folder).resolve()
    with (folder / "training_manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    path, meta_path = folder / "boundary20.joblib", folder / "metadata.json"
    if (manifest.get("version") != VERSION or _sha(path) != manifest.get("model_sha256")
            or _sha(meta_path) != manifest.get("metadata_sha256")):
        raise ValueError("boundary20 model/metadata artifact certificate differs")
    with meta_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    reference, heldout, sources, train, _, bindings, registry_sha = _training_context(prepared, ANCHOR_SERIES)
    _metadata_sources(metadata, ANCHOR_SERIES)
    if (metadata.get("feature_set") != "boundary20" or metadata.get("training_rows") != len(train)
            or metadata.get("training_sources") != [list(s) for s in sources]
            or metadata.get("held_out_identity") != list(heldout)
            or metadata.get("training_regions_sha256") != registry_sha
            or metadata.get("training_source_sha256") != bindings
            or metadata.get("library_versions") != prepared.get("versions", {})
            or metadata.get("training_class_counts") != dict(Counter(train.target_action))
            or metadata.get("sample_weights_sha256") != reference.get("training_weights_sha256",
                reference.get("sample_weights_sha256", metadata.get("sample_weights_sha256")))
            or manifest.get("protected_training_sha256") != bindings
            or manifest.get("training_regions_sha256") != registry_sha
            or manifest.get("held_out_identity") != list(heldout)
            or Path(metadata.get("model_path", "")).resolve() != path
            or metadata.get("model_sha256") != manifest["model_sha256"]
            or metadata.get("feature_bank_version") != _bank_api().VERSION
            or metadata.get("new_monitor_medical_training_data_used") is not False):
        raise ValueError("boundary20 model does not match the authenticated old ten-source fold")
    _check(bindings)
    model = joblib.load(path)
    _validate_model(model, metadata, feature_columns("boundary20"))
    return model, metadata
