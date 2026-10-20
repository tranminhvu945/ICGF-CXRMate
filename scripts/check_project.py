#!/usr/bin/env python3
"""Preflight checks for ICGF-CXRMate (no dataset or checkpoint required).

Usage:
    python scripts/check_project.py
    python scripts/check_project.py --runtime

The --runtime mode requires all packages from requirement.txt to be installed.
The local config/paths.yaml is intentionally ignored (example placeholders are used).
This script does not train, download pretrained models or read clinical records.
"""
import argparse
import ast
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_DIR = ROOT / 'config'
LOCAL_PATHS = CONFIG_DIR / 'paths.yaml'
EXAMPLE_PATHS = CONFIG_DIR / 'paths.example.yaml'


def static_checks():
    python_files = sorted(ROOT.rglob('*.py'))
    # Ignore machine-specific config/paths.yaml even if it is malformed.
    yaml_files = sorted(p for p in CONFIG_DIR.rglob('*.yaml') if p != LOCAL_PATHS)
    json_files = sorted((ROOT / 'cxrmate_ed').glob('*.json'))
    failures = []

    for path in python_files:
        try:
            compile(path.read_bytes(), str(path), 'exec')
        except Exception as exc:
            failures.append(f'Python syntax: {path}: {exc}')

    try:
        import yaml
    except ImportError as exc:
        failures.append('PyYAML is required to parse configurations: ' + str(exc))
        yaml = None

    parsed = {}
    if yaml:
        for path in yaml_files:
            try:
                cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
                if not isinstance(cfg, dict):
                    raise ValueError('expected mapping')
                parsed[path.relative_to(CONFIG_DIR).as_posix()] = cfg
            except Exception as exc:
                failures.append(f'YAML: {path}: {exc}')

    for path in json_files:
        try:
            json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:
            failures.append(f'JSON: {path}: {exc}')

    base = parsed.get('base.yaml', {})
    if base.get('defaults') != ['paths', '_self_']:
        failures.append('config/base.yaml should include paths.yaml at root')
    if not EXAMPLE_PATHS.is_file():
        failures.append('Missing config/paths.example.yaml')

    for rel_path, cfg in parsed.items():
        if rel_path in ('base.yaml', 'paths.example.yaml'):
            continue
        if rel_path.startswith(('ablations/', 'baselines/')):
            path = CONFIG_DIR / rel_path
            if not path.read_text(encoding='utf-8').startswith('# @package _global_'):
                failures.append(f'Missing Hydra root package directive: {rel_path}')
            if '/base' not in cfg.get('defaults', []):
                failures.append(f'Missing absolute base config default: {rel_path}')
        elif 'base' not in cfg.get('defaults', []):
            failures.append(f'Missing base default: {rel_path}')
        if not cfg.get('module') or not cfg.get('definition'):
            failures.append(f'Missing model module/class: {rel_path}')

    lightning_path = ROOT / 'cxrmate_ed' / 'lightning_module.py'
    declared = {n.name for n in ast.parse(lightning_path.read_text()).body if isinstance(n, ast.ClassDef)}
    for rel_path, cfg in parsed.items():
        if cfg.get('module') == 'cxrmate_ed.lightning_module':
            if cfg.get('definition') not in declared:
                failures.append(f'Missing LightningModule class {cfg.get("definition")}: {rel_path}')

    if failures:
        for issue in failures:
            print('[FAIL]', issue)
        return False
    print(f'[PASS] Syntax for {len(python_files)} Python files')
    print(f'[PASS] Parsed {len(yaml_files)} YAML and {len(json_files)} JSON files')
    print('[PASS] All model definitions and config package directives')
    print('[SKIP] Local config/paths.yaml (intentionally excluded from checks)')
    return True



def fallback_config_checks():
    """Validate config inheritance/interpolation without requiring Hydra.

    This is an additional static check, not a substitute for Hydra composition.
    Model module/class and package directives are checked in static_checks().
    """
    try:
        from omegaconf import OmegaConf
    except ImportError:
        print('[SKIP] OmegaConf merge checks (omegaconf is not installed)')
        return True
    try:
        paths = OmegaConf.load(EXAMPLE_PATHS)
        base = OmegaConf.load(CONFIG_DIR / 'base.yaml')
        # defaults are metadata for Hydra, not runtime keys.
        for name in ('stage_1', 'stage_2', 'stage_3'):
            stage = OmegaConf.load(CONFIG_DIR / f'{name}.yaml')
            cfg = OmegaConf.merge(paths, base, stage)
            resolved = OmegaConf.to_container(cfg, resolve=True)
            for key in ('exp_dir', 'database_dir', 'module', 'definition'):
                if key not in resolved:
                    raise ValueError(f'{name}: missing key {key}')
            print(f'[PASS] OmegaConf merge: {name}')
        for folder in ('baselines', 'ablations'):
            for path in sorted((CONFIG_DIR / folder).glob('*.yaml')):
                cfg = OmegaConf.merge(paths, base, OmegaConf.load(path))
                resolved = OmegaConf.to_container(cfg, resolve=True)
                for key in ('exp_dir', 'database_dir', 'module', 'definition'):
                    if key not in resolved:
                        raise ValueError(f'{path.name}: missing key {key}')
                print(f'[PASS] OmegaConf merge: {folder}/{path.stem}')
    except Exception as exc:
        print('[FAIL] OmegaConf merge:', type(exc).__name__, exc)
        return False
    return True

def hydra_checks():
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except ImportError as exc:
        print('[FAIL] Hydra is not importable:', exc)
        return False

    import shutil
    import tempfile
    import yaml

    with tempfile.TemporaryDirectory(prefix='icgf-hydra-') as work:
        work_path = Path(work) / 'config'
        shutil.copytree(CONFIG_DIR, work_path)
        # Always compose against a known-good, portable paths template.
        # Never validate or depend on private machine-specific path values.
        shutil.copy2(work_path / 'paths.example.yaml', work_path / 'paths.yaml')
        paths = ['stage_1', 'stage_2', 'stage_3']
        paths += [p.relative_to(work_path).with_suffix('').as_posix()
                  for folder in ('baselines', 'ablations')
                  for p in sorted((work_path / folder).glob('*.yaml'))]
        try:
            with initialize_config_dir(version_base=None, config_dir=str(work_path)):
                for name in paths:
                    cfg = compose(config_name=name)
                    OmegaConf.to_container(cfg, resolve=True)
                    for key in ('exp_dir', 'database_dir', 'module', 'definition'):
                        if key not in cfg:
                            raise ValueError(f'{name}: missing root-level {key}')
                    if cfg.module != 'cxrmate_ed.lightning_module':
                        raise ValueError(f'{name}: unexpected module={cfg.module}')
                    print(f'[PASS] Hydra compose: {name}')
        except Exception as exc:
            print('[FAIL] Hydra compose:', type(exc).__name__, exc)
            return False
    return True


def import_checks():
    names = [
        'cxrmate_ed.configuration_cxrmate_ed',
        'cxrmate_ed.dataset',
        'cxrmate_ed.utils',
        'cxrmate_ed.create_section_files',
        'cxrmate_ed.section_parser',
        'cxrmate_ed.prepare_dataset',
        'cxrmate_ed.modelling_cxrmate_ed',
        'cxrmate_ed.metrics',
        'cxrmate_ed.rewards',
        'cxrmate_ed.lightning_module',
        'stages',
        'main',
        'scripts.inference',
        'scripts.prepare_dataset',
    ]
    failed = False
    for name in names:
        try:
            importlib.import_module(name)
            print(f'[PASS] import {name}')
        except Exception as exc:
            failed = True
            print(f'[FAIL] import {name}: {type(exc).__name__}: {exc}')
    return not failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', action='store_true',
                        help='Also compose actual Hydra configs and import code (dependencies required)')
    args = parser.parse_args()
    static_ok = static_checks()
    config_ok = fallback_config_checks()
    good = static_ok and config_ok
    if args.runtime:
        # Do not short-circuit: collect diagnostics from both phases.
        hydra_ok = hydra_checks()
        imports_ok = import_checks()
        good = good and hydra_ok and imports_ok
    print('\n' + ('ALL REQUESTED CHECKS PASSED' if good else 'CHECKS FAILED'))
    return 0 if good else 1


if __name__ == '__main__':
    raise SystemExit(main())
