"""Universal Quant Agent Streamlit package.

Shared page helpers are exposed through :mod:`app.page_runtime`. Keeping this
package initializer side-effect free prevents Streamlit navigation pages from
re-importing the running entry script as a transient ``__main__`` module.
"""