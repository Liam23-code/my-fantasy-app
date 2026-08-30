# Local launch: avoiding the `app` package shadow

## The failure

```
ModuleNotFoundError: No module named 'app.style'; 'app' is not a package
```

This appears only for certain **local launch paths**. The repository structure is
correct, `UniversalQuantAgent/app/__init__.py` exists, and the full test suite
(1402 tests) passes. The error is purely about which directory Python treats as
the location of the top-level name `app`.

## Why launching from inside `UniversalQuantAgent/app/` breaks imports

`streamlit run app/app.py` puts the **script's own directory** on `sys.path[0]`.

- Launch from `UniversalQuantAgent/` &rarr; `sys.path[0]` is
  `UniversalQuantAgent/app`. There is no directory named `app` inside
  `UniversalQuantAgent/app`, so `import app` keeps searching and finds the real
  `UniversalQuantAgent/app/` **package** (via the project root, which
  `app/app.py` adds to `sys.path`). `from app.style import ...` works.
- Launch from `UniversalQuantAgent/app/` &rarr; `sys.path[0]` is
  `UniversalQuantAgent/app` **and** that is also the current working directory.
  Now `import app` finds `app.py` in that directory first and binds the name
  `app` to a plain **module**. `app.style` then fails with
  `'app' is not a package`, because a module has no submodules.

The same thing happens with `python -c "from app.style import ..."` or any
`import app` run from within `UniversalQuantAgent/app/`: the module is cached in
`sys.modules["app"]` as a non-package before the path fix in `app/app.py` runs.

## Why launching from the repo root works

From the repo root, `streamlit run UniversalQuantAgent/app/app.py` sets
`sys.path[0]` to `UniversalQuantAgent/app`, and `app/app.py` immediately inserts
`UniversalQuantAgent/` at the front of `sys.path`. `import app` resolves to the
real package. This is also what the Render deployment does
(`render.yaml` &rarr; `streamlit run UniversalQuantAgent/app/app.py`), which is why
production has never seen this error.

## The guard in `app/app.py`

Before the first `from app...` import, `app/app.py` now:

1. Removes every stale copy of the project-root path from `sys.path`, then
   re-inserts it at index 0, so the real package is always found first.
2. Deletes `sys.modules["app"]` **only if** it is a non-package shadow
   (`not hasattr(module, "__path__")`). A correctly imported `app` package has
   `__path__` and is left untouched.

Every page under `app/pages/` already carries the same guard, so navigating
between pages cannot reintroduce the shadow.

## `run_local.py`

`run_local.py` at the repo root is the recommended way to start the app locally:

```bash
python run_local.py
# pass-through args work too:
python run_local.py --server.port 8600
```

It works no matter which directory you run it from because it:

- resolves the repo root from its own location and `chdir`s there (so Streamlit
  also picks up `.streamlit/config.toml`, matching Render);
- forces the repo root and `UniversalQuantAgent/` to the front of `sys.path`;
- purges a shadowed non-package `app` from `sys.modules`;
- then hands off to `streamlit run UniversalQuantAgent/app/app.py`.

## TL;DR

| Launch method | Result |
| --- | --- |
| `python run_local.py` (any cwd) | works |
| `streamlit run UniversalQuantAgent/app/app.py` from repo root | works |
| `streamlit run app/app.py` from `UniversalQuantAgent/` | works |
| `streamlit run app.py` / `import app` from `UniversalQuantAgent/app/` | shadow &rarr; guarded by `app/app.py`; use `run_local.py` instead |
