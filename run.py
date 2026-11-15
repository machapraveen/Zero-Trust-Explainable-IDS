#!/usr/bin/env python
"""
ZT-XAIIDS Application Entry Point
==================================
Explainable Deep Learning Framework for Zero-Trust Intrusion Detection
Using Multi-Level Network Traffic Analytics
Department of CSE - AIML | Aditya College of Engineering and Technology
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import create_app


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='ZT-XAIIDS - Explainable Deep Learning Framework for Zero-Trust Intrusion Detection'
    )
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--production', action='store_true', help='Run in production mode')
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.production:
        os.environ['FLASK_ENV'] = 'production'
    else:
        os.environ['FLASK_ENV'] = os.environ.get('FLASK_ENV', 'development')

    app = create_app()
    print("=" * 72)
    print("  ZT-XAIIDS | Explainable Deep Learning Zero-Trust Intrusion Detection")
    print("  Aditya College of Engineering and Technology (Dept. of CSE - AIML)")
    print("=" * 72)
    print(f"  URL:            http://{args.host}:{args.port}")
    print(f"  Zero-Trust Lab: http://{args.host}:{args.port}/zero-trust")
    print("=" * 72)

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug or (not args.production),
        threaded=True,
    )


if __name__ == '__main__':
    main()
