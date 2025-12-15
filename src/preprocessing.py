"""
`preprocessing.py` — thin compatibility wrapper.
Re-exports the main `FeatureEngineer` class or `run_feature_engineering` function
from the existing `preprocessing_file.py` implementation.
"""
try:
    from preprocessing_file import FeatureEngineer

    def run_preprocessing(*args, **kwargs):
        fe = FeatureEngineer()
        if hasattr(fe, 'run_feature_engineering'):
            return fe.run_feature_engineering(*args, **kwargs)
        if hasattr(fe, 'run'):
            return fe.run(*args, **kwargs)
        raise RuntimeError('No run entrypoint found on FeatureEngineer')

except Exception:
    def run_preprocessing(*args, **kwargs):
        raise RuntimeError('Preprocessing implementation not available (missing preprocessing_file.py)')

if __name__ == '__main__':
    run_preprocessing()
