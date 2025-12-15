"""
Thin wrapper module `etl.py` that exposes a simple `run_etl()` entrypoint.
It delegates to the existing `etl_pipeline.py` implementation when present.
"""
from typing import Any

try:
    # primary implementation
    from etl_pipeline import PredictiveMaintenanceETL

    def run_etl(*args: Any, **kwargs: Any) -> None:
        """Run the project's ETL pipeline (delegates to `etl_pipeline`)."""
        etl = PredictiveMaintenanceETL()
        # many implementations expose run_pipeline or run
        if hasattr(etl, 'run_pipeline'):
            return etl.run_pipeline(*args, **kwargs)
        if hasattr(etl, 'run'):
            return etl.run(*args, **kwargs)
        # fallback: try a method named execute
        if hasattr(etl, 'execute'):
            return etl.execute(*args, **kwargs)
        raise RuntimeError('No runnable ETL entrypoint found on PredictiveMaintenanceETL')

except Exception:
    # graceful fallback for imports (module may already follow new layout)
    def run_etl(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError('ETL implementation not available (missing etl_pipeline.py)')


if __name__ == '__main__':
    run_etl()
