#!/usr/bin/env python3
"""Copy editable Voyager skill packages into a checkout's .agents/skills.

Standard library only. Sources stay under docs/skills; installed copies are
independent files whose relative links refer to the destination checkout.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit

NAMES = ('voyager-compiler-architecture', 'voyager-target-adoption')


def install(repo, *, replace=False, dry_run=False):
    source_root = Path(__file__).resolve().parent
    repo = Path(repo).expanduser().resolve()
    if not (repo / 'docs/README.md').is_file():
        raise ValueError('Destination must be a checkout containing docs/README.md')
    destination = repo / '.agents/skills'
    if not destination.resolve().is_relative_to(repo):
        raise ValueError('Destination .agents/skills resolves outside the checkout')
    actions = []
    # Preflight both packages before making any changes.
    for name in NAMES:
        source = source_root / name
        target = destination / name
        if target.is_symlink():
            raise ValueError(f'Refusing to replace a symlink: {target}')
        if target.exists() and (not target.is_dir() or not replace):
            raise ValueError(f'Destination exists: {target}; use --replace to replace its local edits')
        if not (source / 'SKILL.md').is_file():
            raise ValueError(f'Missing source skill: {source}')
        for path in source.rglob('*'):
            if path.is_symlink():
                raise ValueError(f'Skill sources must be independent files, not symlinks: {path}')
            if path.suffix != '.md':
                continue
            for href in re.findall(r'\]\(([^)]+)\)', path.read_text()):
                url = urlsplit(href)
                if url.scheme or url.netloc:
                    continue
                if Path(unquote(url.path)).is_absolute():
                    raise ValueError(f'Absolute link in skill: {path}: {href}')
                staged_file = target / path.relative_to(source)
                resolved = (staged_file.parent / unquote(url.path)).resolve()
                if not resolved.is_relative_to(repo):
                    raise ValueError(f'Link leaves destination checkout: {href}')
                if resolved.is_relative_to(target.resolve()):
                    available = source / resolved.relative_to(target.resolve())
                else:
                    available = resolved
                if not available.exists():
                    raise ValueError(f'Missing destination document for {href}: {available}')
        actions.append((source, target))
    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)
        for source, target in actions:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
    return dict(dry_run=dry_run, checkout=str(repo), installed=[str(t.relative_to(repo)) for _, t in actions], source='docs/skills', global_skills_changed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2], help='Destination checkout; defaults to the checkout containing this script')
    parser.add_argument('--replace', action='store_true', help='Replace both installed skill directories, including edits made there; sources under docs/skills remain unchanged')
    parser.add_argument('--dry-run', action='store_true', help='Validate paths and print destinations without writing')
    args = parser.parse_args()
    try:
        result = install(args.repo, replace=args.replace, dry_run=args.dry_run)
    except (ValueError, OSError) as error:
        parser.exit(1, f'{error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
