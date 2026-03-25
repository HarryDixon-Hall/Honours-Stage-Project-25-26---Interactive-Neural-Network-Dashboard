from importlib import import_module

import dash
from dash import html
try:
    from dash import Input, Output
except ImportError:
    dash_dependencies = import_module('dash.dependencies')
    Input = dash_dependencies.Input
    Output = dash_dependencies.Output
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.decomposition import PCA

from dataload import load_dataset
from models import SimpleNN