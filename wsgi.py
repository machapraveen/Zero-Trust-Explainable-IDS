"""
WSGI Entry Point for ZT-XAIIDS
==============================
Production Gunicorn / Waitress WSGI entrypoint.
"""

import os
from app import create_app

application = create_app(os.environ.get('FLASK_ENV', 'production'))
app = application
