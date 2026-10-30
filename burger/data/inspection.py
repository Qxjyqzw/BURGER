"""CPU-only validation of the LIBERO-style training HDF5 schema."""
from pathlib import Path
import json
import h5py
import numpy as np

LIBERO_ROOT_ATTRS = ["bddl_file_name", "env_args", "env_name", "macros_image_convention", "num_demos", "problem_info", "tag", "total"]
LIBERO_DEMO_ATTRS = ["init_state", "model_file", "num_samples"]
LIBERO_DEMO_DATASETS = {
    "actions": ((7,), "float64"), "dones": ((), "uint8"), "rewards": ((), "uint8"),
    "robot_states": ((9,), "float64"), "states": (None, "float64"),
    "obs/agentview_rgb": ((None, None, 3), "uint8"), "obs/eye_in_hand_rgb": ((None, None, 3), "uint8"),
    "obs/ee_ori": ((3,), "float64"), "obs/ee_pos": ((3,), "float64"), "obs/ee_states": ((6,), "float64"),
    "obs/gripper_states": ((2,), "float64"), "obs/joint_states": ((7,), "float64"),
}


def _leaf_datasets(grp, prefix=""):
    out = {}
    for k in grp.keys():
        it = grp[k]
        if isinstance(it, h5py.Dataset):
            out[prefix + k] = it
        else:
            out.update(_leaf_datasets(it, prefix + k + "/"))
    return out


def check_structure(path: str, reference: str | None) -> bool:
    """Return False for malformed files, including missing datasets and empty episodes."""
    try:
        return _check_structure(path, reference)
    except (KeyError, ValueError, TypeError, IndexError, OSError) as exc:
        print(f"[structure] FAIL: {exc}")
        return False


def _check_structure(path: str, reference: str | None) -> bool:
    ok = True
    with h5py.File(path, "r") as f:
        if "data" not in f:
            print("[structure] FAIL: missing data group")
            return False
        data = f["data"]
        for a in LIBERO_ROOT_ATTRS:
            if a not in data.attrs:
                print(f"  MISSING root attr: {a}"); ok = False
        if not ok:
            return False
        if not len(data):
            print("[structure] FAIL: no demonstrations")
            return False
        print(f"[structure] {path}: {data.attrs['num_demos']} demos, {data.attrs['total']} steps")
        env_args = json.loads(data.attrs["env_args"])
        ek = env_args.get("env_kwargs", {})
        ctrl = ek.get("controller_configs", {}).get("type", ek.get("control_mode", "n/a"))
        print(f"  env_name={env_args.get('env_name')} problem_name={env_args.get('problem_name')} controller={ctrl} "
              f"cameras={ek.get('camera_names', 'n/a')} {ek.get('camera_heights', ek.get('camera_height', '?'))}x{ek.get('camera_widths', ek.get('camera_width', '?'))}")
        print(f"  language: {json.loads(data.attrs['problem_info'])['language_instruction']!r} | image convention: {data.attrs['macros_image_convention']}")
        extras = set()
        total = 0
        generation = json.loads(data.attrs.get("painter_generation", "{}"))
        native = generation.get("action_mode") == "native"
        action_dim = int(generation.get("action_space", {}).get("dim", 7)) if native else 7
        bounded_actions = not native or generation.get("action_space", {}).get("type") == "libero_osc_pose_delta"
        for demo_name in sorted(data.keys(), key=lambda s: int(s.split("_")[1])):
            demo = data[demo_name]
            for a in LIBERO_DEMO_ATTRS:
                if a not in demo.attrs:
                    print(f"  {demo_name}: MISSING attr {a}"); ok = False
            if "num_samples" not in demo.attrs:
                continue
            leaves = _leaf_datasets(demo)
            T = int(demo.attrs["num_samples"])
            if T <= 0:
                print(f"  {demo_name}: empty demonstration"); ok = False; continue
            total += T
            for name, (suffix, dtype) in LIBERO_DEMO_DATASETS.items():
                if name == "actions":
                    suffix = (action_dim,)
                if name not in leaves:
                    print(f"  {demo_name}: MISSING dataset {name}"); ok = False; continue
                ds = leaves[name]
                if not ds.shape or ds.shape[0] != T:
                    print(f"  {demo_name}: {name} shape {ds.shape}, expected {T} rows"); ok = False
                if suffix is not None and len(ds.shape) != 1 + len(suffix):
                    print(f"  {demo_name}: {name} rank {len(ds.shape)} != {1 + len(suffix)}"); ok = False
                if suffix is not None:
                    for want, got in zip(suffix, ds.shape[1:]):
                        if want is not None and want != got:
                            print(f"  {demo_name}: {name} shape {ds.shape} does not match suffix {suffix}"); ok = False
                if str(ds.dtype) != dtype:
                    print(f"  {demo_name}: {name} dtype {ds.dtype} != {dtype}"); ok = False
                if np.issubdtype(ds.dtype, np.floating) and not np.isfinite(ds[:]).all():
                    print(f"  {demo_name}: {name} contains NaN or infinity"); ok = False
            if "actions" in leaves and leaves["actions"].size:
                acts = demo["actions"][:]
                if bounded_actions and (acts.min() < -1.0 - 1e-6 or acts.max() > 1.0 + 1e-6):
                    print(f"  {demo_name}: actions outside [-1, 1]: {acts.min():.3f}..{acts.max():.3f}"); ok = False
            for key in ("dones", "rewards"):
                if key in leaves and leaves[key].shape == (T,) and int(leaves[key][-1]) != 1:
                    print(f"  {demo_name}: last {key} should be 1"); ok = False
            extras.update(k for k in leaves if k not in LIBERO_DEMO_DATASETS)
        if len(data) != int(data.attrs["num_demos"]) or total != int(data.attrs["total"]):
            print("  root episode/step totals differ from the stored demonstrations"); ok = False
        if extras:
            print(f"  extra painter datasets (ignored by LIBERO loaders): {sorted(extras)}")
    if reference:
        with h5py.File(reference, "r") as rf, h5py.File(path, "r") as f:
            rd = rf["data"][sorted(rf["data"].keys())[0]]; d = f["data"]["demo_0"]
            rl, l = _leaf_datasets(rd), _leaf_datasets(d)
            missing = [k for k in rl if k not in l]
            print(f"  reference {Path(reference).name}: datasets missing vs reference: {missing or 'none'}")
            for k in rl:
                if k in l:
                    same = (rl[k].shape[1:] == l[k].shape[1:] or k in ("states", "obs/agentview_rgb", "obs/eye_in_hand_rgb")) and str(rl[k].dtype) == str(l[k].dtype)
                    if not same:
                        print(f"    {k}: reference shape {rl[k].shape} {rl[k].dtype} vs ours {l[k].shape} {l[k].dtype}"); ok = False
            ra = set(rf["data"].attrs.keys()); oa = set(f["data"].attrs.keys())
            print(f"  root attrs missing vs reference: {sorted(ra - oa) or 'none'}; extra: {sorted(oa - ra) or 'none'}")
            print(f"  demo attrs missing vs reference: {sorted(set(rd.attrs.keys()) - set(d.attrs.keys())) or 'none'}")
    print(f"[structure] {'PASS' if ok else 'FAIL'}")
    return ok
