#!/usr/bin/env python3
"""
TARA Universal Model - Simple Training Monitor
Provides a clean, efficient dashboard for monitoring training progress
"""

import os
import sys
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify, Response 