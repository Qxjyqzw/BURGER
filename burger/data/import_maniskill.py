#!/usr/bin/env python
"""Convert existing painter ManiSkill H5 + sidecars to LIBERO-style training HDF5.

CPU only. Keeps the source read-only; does not instantiate either simulator.
The resulting ManiSkill-derived data cannot be replayed in MuJoCo.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

import h5py
import numpy as np

from burger._paths import PROJECT_ROOT as REPO
from burger.data import canonical as C
from burger.data.exporters.writers import write_libero_hdf5
from burger.data.inspection import check_structure


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def require(condition, message):
    """Input and conversion checks must still run under python -O."""
    if not condition:
        raise ValueError(message)


def validate(source, output, selected, base, hw, gain):
    """Check all low-dimensional rows and independently sample image alignment."""
    from PIL import Image
    require(check_structure(str(output), None), 'Repository schema check failed')
    frame_checks = 0
    max_action_error = 0.0
    clipped = 0
    components = 0
    with h5py.File(source, 'r') as src, h5py.File(output, 'r') as dst:
        data = dst['data']
        require(len(data) == len(selected), 'validation failed: len(data) == len(selected)')
        total = 0
        for i, name in enumerate(selected):
            s, d = src[name], data[f'demo_{i}']
            a = s['actions'][:].astype(np.float64)
            n = len(a)
            tcp = s['obs/extra/tcp_pose'][:n].astype(np.float64)
            actual = d['actions'][:]
            raw_delta = gain * (a[:, :3] + base - tcp[:, :3]) / 0.05
            expected = np.column_stack((np.clip(raw_delta, -1, 1), np.zeros((n, 3)), np.ones(n)))
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
            max_action_error = max(max_action_error, float(np.max(np.abs(actual - expected))))
            np.testing.assert_array_equal(d['obs/ee_pos'][:], tcp[:, :3])
            xyzw = tcp[:, [4, 5, 6, 3]]
            np.testing.assert_array_equal(d['robot_states'][:, 5:], xyzw)
            # Reconstruct orientation from axis-angle, allowing quaternion sign equivalence.
            from scipy.spatial.transform import Rotation
            recovered = Rotation.from_rotvec(d['obs/ee_ori'][:]).as_quat()
            q = xyzw / np.linalg.norm(xyzw, axis=1, keepdims=True)
            np.testing.assert_allclose(np.abs(np.sum(recovered * q, axis=1)), 1, atol=2e-6)
            require(d['states'].shape == (n, 0), "validation failed: d['states'].shape == (n, 0)")
            require(not d.attrs['model_file'], "validation failed: not d.attrs['model_file']")
            require(d.attrs['maniskill_traj'] == name, "validation failed: d.attrs['maniskill_traj'] == name")
            require(d['obs/agentview_rgb'].shape == (n, hw[0], hw[1], 3), "validation failed: d['obs/agentview_rgb'].shape == (n, hw[0], hw[1], 3)")
            for key in ['joint_states', 'gripper_states']:
                require(not np.any(d['obs'][key][:]), "validation failed: not np.any(d['obs'][key][:])")
            for key in ['actions', 'robot_states', 'obs/ee_ori', 'obs/ee_states']:
                require(np.isfinite(d[key][:]).all(), (name, key))
            for key in ['rewards', 'dones']:
                arr = d[key][:]
                require(arr[-1] == 1 and not np.any(arr[:-1]), 'validation failed: arr[-1] == 1 and not np.any(arr[:-1])')
            for t in sorted({0, n // 2, n - 1}):
                original = s['obs/sensor_data/base_camera/rgb'][t]
                resized = np.asarray(Image.fromarray(original).resize((hw[1], hw[0]), Image.Resampling.BILINEAR))
                np.testing.assert_array_equal(d['obs/agentview_rgb'][t], resized[::-1])
                require(not np.any(d['obs/eye_in_hand_rgb'][t]), "validation failed: not np.any(d['obs/eye_in_hand_rgb'][t])")
                frame_checks += 1
            clipped += int((np.abs(raw_delta) > 1).sum())
            components += raw_delta.size
            total += n
        require(total == int(data.attrs['total']), "validation failed: total == int(data.attrs['total'])")
        require(int(data.attrs['num_demos']) == len(selected), "validation failed: int(data.attrs['num_demos']) == len(selected)")
    return dict(passed=True, schema_check=True, all_low_dimensional_rows_checked=total,
                sampled_image_frames_checked=frame_checks, max_action_error=max_action_error,
                translation_component_clipping_fraction=clipped / components,
                simulator_replay=False, actual_training_run=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input_dir', type=Path, required=True)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--robot_base_world', type=float, nargs=3, required=True,
                   help='Actual runtime robot base, not the summary initial_pose placeholder')
    p.add_argument('--image_hw', type=int, nargs=2, default=[128, 128])
    p.add_argument('--delta_gain', type=float, default=4)
    p.add_argument('--compression', choices=['lzf', 'gzip', 'none'], default='lzf')
    p.add_argument('--max_episodes', type=int)
    p.add_argument('--allow_unreviewed_summary_selection', action='store_true',
                   help='Explicitly trust dataset_summary parallel-tail counts without a filter_manifest.json; no quality review is implied')
    args = p.parse_args(argv)
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    require(np.isfinite(args.robot_base_world).all(), 'robot_base_world must be finite')
    require(min(args.image_hw) > 0, 'image_hw must be positive')
    require(np.isfinite(args.delta_gain) and args.delta_gain > 0, 'delta_gain must be positive and finite')
    started = time.perf_counter()
    started_utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    base = np.asarray(args.robot_base_world, dtype=np.float64)
    inputs = sorted(args.input_dir.glob('*.h5'))
    if len(inputs) != 1:
        raise ValueError(f'Expected one source .h5, got {inputs}')
    source = inputs[0]
    before = (source.stat().st_size, source.stat().st_mtime_ns)
    meta_path = source.with_suffix('.json')
    summary_path = args.input_dir / 'dataset_summary.json'
    manifest_path = args.input_dir / 'filter_manifest.json'
    map_path = source.with_name(source.stem + '_traj_to_svg.json')
    meta = json.loads(meta_path.read_text())
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest is None and not args.allow_unreviewed_summary_selection:
        raise ValueError('A reviewed filter_manifest.json is required. If summary-only episode selection is intentional, '
                         'pass --allow_unreviewed_summary_selection; this does not verify drawing quality.')
    mapping = json.loads(map_path.read_text())['traj_to_svg'] if map_path.exists() else {}
    ek = meta['env_info']['env_kwargs']
    if meta['env_info']['env_id'] != 'DrawSVGParallel-v1' or ek['control_mode'] != 'pd_ee_pose':
        raise ValueError('Only painter DrawSVGParallel-v1 absolute pd_ee_pose is supported')
    language = summary.get('source_desc') or meta.get('source_desc')
    if not language or len(summary['objects']) != 1:
        raise ValueError('Expected a single task with a language instruction')
    object_name = summary['objects'][0]
    require(isinstance(object_name, str) and object_name not in ('', '.', '..')
            and Path(object_name).name == object_name and '/' not in object_name and '\\' not in object_name,
            'summary object name must be a single folder/file component')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    canonical = out / f'{object_name}.canonical.h5'
    final = out / f'{object_name}_demo.hdf5'
    for f in [canonical, final, out / 'conversion_report.json']:
        if f.exists():
            raise FileExistsError(f'Will not overwrite {f}')
    with h5py.File(source, 'r') as src:
        names = sorted(src.keys(), key=lambda x: int(x.split('_')[1]))
        require(names == [f'traj_{i}' for i in range(len(names))], "validation failed: names == [f'traj_{i}' for i in range(len(names))]")
        tail = int(summary['num_envs']) - int(summary['last_meaningful_env_num'])
        require(0 <= tail < int(summary['num_envs']), 'invalid parallel-tail counts in dataset_summary.json')
        if manifest is not None:
            require(int(manifest['trajectory_count_after']) == len(names), "validation failed: int(manifest['trajectory_count_after']) == len(names)")
            require(int(manifest['tail_trajectory_count']) == tail, "validation failed: int(manifest['tail_trajectory_count']) == tail")
            valid_count = int(manifest['reviewed_count_after'])
            require(valid_count + tail == len(names), 'validation failed: valid_count + tail == len(names)')
        else:
            valid_count = len(names) - tail
        require(0 < valid_count <= len(names), 'validation failed: 0 < valid_count <= len(names)')
        selected = names[:valid_count]
        if args.max_episodes is not None:
            require(args.max_episodes > 0, 'validation failed: args.max_episodes > 0')
            selected = selected[:args.max_episodes]
        for name in selected:
            g = src[name]
            a = g['actions'][:]
            n = len(a)
            require(a.shape == (n, 6) and n > 0, 'validation failed: a.shape == (n, 6) and n > 0')
            require(np.isfinite(a).all(), 'validation failed: np.isfinite(a).all()')
            np.testing.assert_allclose(a[:, 3:], np.broadcast_to([np.pi, 0., 0.], (n, 3)), atol=1e-5)
            require(g['obs/extra/tcp_pose'].shape == (n + 1, 7), "validation failed: g['obs/extra/tcp_pose'].shape == (n + 1, 7)")
            require(g['obs/sensor_data/base_camera/rgb'].shape[0] == n + 1, "validation failed: g['obs/sensor_data/base_camera/rgb'].shape[0] == n + 1")
            require(bool(g['success'][-1]), f'{name} is not a successful demonstration')
            if 'wrist_camera' in g['obs/sensor_data']:
                raise ValueError('This entry point currently expects the single-camera painter dataset')
        total = sum(len(src[n]['actions']) for n in selected)
    selection = dict(raw_episodes=len(names), selected_candidate_episodes=valid_count,
                     selection_source='filter_manifest' if manifest is not None else 'unreviewed_dataset_summary',
                     quality_reviewed=manifest is not None,
                     excluded_parallel_tail=names[valid_count:], selected=selected,
                     max_episodes=args.max_episodes, total_steps=total)
    dump(out / 'source_selection.json', selection)
    kw = {k: v for k, v in ek.items() if k != 'svg_string_list'}
    info = C.DatasetInfo(
        backend='maniskill', language=language,
        control_freq=int(summary['env_config']['simulation_config']['control_freq']),
        robot='panda_stick', cameras=['agentview_rgb'],
        action_space=dict(type='maniskill_pd_ee_pose_abs', dim=6,
                          robot_base_world=base.tolist(), description='Absolute robot-root xyz + Euler xyz'),
        backend_config=dict(env_name='DrawSVGParallel-v1', problem_name='maniskill_draw_svg_parallel',
                            env_kwargs=kw, env_config=summary['env_config'], maniskill_h5=str(source)),
        generation=dict(operation='existing_maniskill_h5_to_libero', input=str(source),
                        robot_base_world=base.tolist(), selection=selection,
                        image_hw=args.image_hw, resize='Pillow bilinear, full frame; changes 4:3 to 1:1',
                        no_simulation=True, missing_camera_policy='zero-fill and flag'))
    print(f'[selection] {len(selected)}/{len(names)} episodes, {total} steps; exclude {tail} parallel tail trajectories', flush=True)
    import_start = time.perf_counter()
    inverse_ids = {int(v): int(k) for k, v in (manifest or {}).get('old_to_new_trajectory_id', {}).items()}
    with h5py.File(source, 'r') as src, C.CanonicalWriter(canonical, info) as writer:
        for i, name in enumerate(selected):
            g = src[name]
            acts = g['actions'][:].astype(np.float64)
            n = len(acts)
            tcp = g['obs/extra/tcp_pose'][:n].astype(np.float64)
            require(np.isfinite(tcp).all(), 'validation failed: np.isfinite(tcp).all()')
            np.testing.assert_allclose(np.linalg.norm(tcp[:, 3:7], axis=1), 1., atol=1e-5)
            rgb = C.resize_images(g['obs/sensor_data/base_camera/rgb'][:n], tuple(args.image_hw))
            tid = int(name.split('_')[1])
            svg_idx = int(g['obs/episode_svg_index'][0, 0])
            m = mapping.get(str(tid), {})
            require(not m or int(m['svg_index']) == svg_idx, "validation failed: not m or int(m['svg_index']) == svg_idx")
            metadata = dict(maniskill_traj=name, source_trajectory_id=tid,
                            original_trajectory_id=inverse_ids.get(tid, tid), svg_index=svg_idx,
                            svg_file=m.get('svg_filename', ''),
                            svg_string=ek['svg_string_list'][svg_idx], success=True)
            ep = C.Episode(obs=dict(agentview_rgb=rgb, eef_pos=tcp[:, :3], eef_quat_wxyz=tcp[:, 3:7],
                                   pen_tip_world=tcp[:, :3], pen_tip_painter=tcp[:, :3] - base),
                           actions=acts, arrays=dict(targets_world=acts[:, :3] + base), meta=metadata)
            # Use source trajectory order, not SVG index (SVG IDs may repeat).
            writer.add_episode(ep, index=i)
            if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(selected):
                print(f'[import] {i+1}/{len(selected)}, {time.perf_counter()-import_start:.2f}s', flush=True)
    import_seconds = time.perf_counter() - import_start
    export_start = time.perf_counter()
    result = write_libero_hdf5(canonical, final, action_mode='libero_delta', delta_gain=args.delta_gain,
                              image_compression=None if args.compression == 'none' else args.compression)
    export_seconds = time.perf_counter() - export_start
    conversion_seconds = time.perf_counter() - started
    print(f'[export] {export_seconds:.2f}s; total conversion {conversion_seconds:.2f}s', flush=True)
    validation_start = time.perf_counter()
    checks = validate(source, final, selected, base, tuple(args.image_hw), args.delta_gain)
    validation_seconds = time.perf_counter() - validation_start
    # Mark the actual provenance explicitly; this is not a MuJoCo replay dataset.
    with h5py.File(final, 'r+') as dst:
        d = dst['data']
        d.attrs['painter_simulator_replayable'] = False
        d.attrs['painter_action_semantics'] = f'derived OSC-style delta, gain={args.delta_gain}; zero rotation, closed gripper; not native MuJoCo actions'
        d.attrs['painter_control_freq'] = info.control_freq
        d.attrs['painter_conversion_source'] = str(source)
        d.attrs['painter_robot_base_world'] = base
    require((source.stat().st_size, source.stat().st_mtime_ns) == before, 'Source changed during conversion')
    versions = {n: importlib.metadata.version(n) for n in ['numpy', 'h5py', 'Pillow', 'scipy']}
    report = dict(status='passed', started_utc=started_utc, input=str(source), output=str(final),
                  canonical=str(canonical), language=language, selection=selection,
                  image_hw=args.image_hw, image_compression=args.compression,
                  robot_base_world=base.tolist(), delta_gain=args.delta_gain,
                  missing=result['missing'], gpu_used=False, cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
                  python=sys.executable, python_version=platform.python_version(), packages=versions,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  timings_seconds=dict(import_resize_canonical=import_seconds, export_libero=export_seconds,
                                       total_conversion=conversion_seconds, validation=validation_seconds,
                                       total_including_validation=time.perf_counter()-started),
                  throughput=dict(steps_per_second=total/conversion_seconds,
                                  episodes_per_second=len(selected)/conversion_seconds,
                                  seconds_per_episode=conversion_seconds/len(selected)),
                  size_bytes=dict(source=before[0], canonical=canonical.stat().st_size, libero=final.stat().st_size),
                  validation=checks, source_size_mtime_unchanged=True,
                  limitations=['ManiSkill visuals and state, approximate OSC-style action mapping.',
                               'No MuJoCo model/state, wrist images, joint or gripper measurements.',
                               'Training loader integration must account for missing fields; no training run performed.'])
    dump(out / 'conversion_report.json', report)
    print(json.dumps({k: report[k] for k in ['status', 'output', 'timings_seconds', 'throughput', 'size_bytes', 'validation']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
