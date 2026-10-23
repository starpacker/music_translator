"""Adapter: let SumisoraOMR's homr_runner talk to the *official* pip `homr` package.

The repo (core/omr/homr_runner.py) was written against the Tsukamotoshio/homr
fork, whose ProcessingConfig carries extra fields (segnet_batch_size,
title_detection). The official homr on PyPI has a narrower dataclass and will
raise TypeError on those kwargs. Installing this shim before homr.main is
imported makes the repo's call sites work unchanged.

Import-order requirement: call install() BEFORE `import homr.main`.
"""
from __future__ import annotations

import dataclasses
import inspect

# Fork-only fields that the official homr does not understand. Mapped to the
# nearest official equivalent where one exists.
_FORK_ONLY = {
    'segnet_batch_size': None,   # no equivalent; batch size is chosen internally
    'title_detection': None,     # no equivalent; title detection is always on
}


def install(verbose: bool = True) -> bool:
    """Patch homr.main.ProcessingConfig to accept (and drop) fork-only kwargs.

    Returns True when the official package needed patching, False when it
    already accepts the repo's kwargs (e.g. a real fork build is installed).
    """
    try:
        import homr.main as homr_main
    except Exception as exc:  # pragma: no cover
        if verbose:
            print(f'[shim] cannot import homr.main: {exc}')
        return False

    original = homr_main.ProcessingConfig
    official_fields = {f.name for f in dataclasses.fields(original)}

    # Already compatible (a genuine fork install) -> leave it alone.
    if {'segnet_batch_size', 'title_detection'} <= official_fields:
        if verbose:
            print('[shim] homr.ProcessingConfig already fork-compatible; no patch needed.')
        return False

    positional = [f.name for f in dataclasses.fields(original)]

    class ProcessingConfig(original):  # type: ignore[misc,valid-type]
        """Official ProcessingConfig + tolerance for fork-only kwargs."""

        def __init__(self, *args, **kwargs):
            dropped = []
            for name in list(kwargs):
                if name in _FORK_ONLY and name not in official_fields:
                    kwargs.pop(name)
                    dropped.append(name)
            # Positional args map 1:1 onto the official field order.
            bound = dict(zip(positional, args))
            bound.update(kwargs)
            for name, default in (
                ('enable_debug', False), ('enable_cache', False),
                ('write_staff_positions', False), ('read_staff_positions', False),
                ('selected_staff', -1), ('transformer_use_gpu', False),
                ('segnet_use_gpu', False), ('coreml_encoder', False),
            ):
                if name in official_fields:
                    bound.setdefault(name, default)
            unknown = set(bound) - official_fields
            for name in unknown:
                bound.pop(name)
            self.__dict__['_dropped_fork_kwargs'] = dropped
            super(original, self).__init__()  # no-op, keeps dataclass happy
            for name, value in bound.items():
                object.__setattr__(self, name, value)

        def __repr__(self) -> str:
            shown = ', '.join(f'{f}={getattr(self, f, None)!r}' for f in positional)
            return f'ProcessingConfig({shown})'

    ProcessingConfig.__name__ = original.__name__
    ProcessingConfig.__qualname__ = original.__qualname__
    ProcessingConfig.__module__ = original.__module__
    # dataclasses.fields() must still work for any downstream introspection.
    dataclasses.dataclass(ProcessingConfig)

    homr_main.ProcessingConfig = ProcessingConfig
    if verbose:
        print(f'[shim] patched homr.ProcessingConfig '
              f'(official fields: {sorted(official_fields)}); '
              f'dropping {sorted(_FORK_ONLY)}')
    return True


if __name__ == '__main__':
    install()
