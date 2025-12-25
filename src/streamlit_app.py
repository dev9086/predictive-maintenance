"""Entry point for Hugging Face Spaces.

This simply re-exports the existing dashboard so that Spaces can load
`streamlit_app.py` as the app file without changing the main code.
"""

from streamlit_dashboard import *  # noqa: F401,F403
