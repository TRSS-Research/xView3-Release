import cv2
import os
import numpy as np
from pandas import DataFrame, CategoricalDtype

resampling_dict = {
    'bilinear': cv2.INTER_LINEAR,
    'bilinear-exact': cv2.INTER_LINEAR_EXACT,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4,
    'cubic': cv2.INTER_CUBIC,
    'max': cv2.INTER_MAX,
    'nearest': cv2.INTER_NEAREST,
    'nearest-exact': cv2.INTER_NEAREST_EXACT,
}
try:
    from rasterio.enums import Resampling
    rasterio_resampling_dict = {
        'bilinear': Resampling.bilinear,
        'bilinear-exact': Resampling.bilinear,
        'area': Resampling.bilinear,
        'lanczos': Resampling.lanczos,
        'cubic': Resampling.cubic,
        'max': Resampling.max,
        'nearest': Resampling.nearest,
        'nearest-exact': Resampling.nearest,
    }
except (ModuleNotFoundError, ImportError):
    pass
#CUSTOM TYPES

TrainData = np.ndarray # 4 dimensional
TestData = np.ndarray # 1 Dimensional
MetadataDataFrame = DataFrame
Centroids = np.ndarray # ((row, col),...)

# static text for convienience
CFAR = 'cfar'
SEGMENTER = 'segment'



performance_output_path = 'evaluation/cfar/performance.csv'
os.makedirs(os.path.dirname(performance_output_path), exist_ok=True)
metadata_centroid_columns = ['detect_scene_row', 'detect_scene_column']
supplemental_data_names = ['mask', 'bathymetry', 'owiWindDirection', 'owiWindQuality', 'owiWindSpeed', ]

data_names_ordered = ['VV', 'VH'] + supplemental_data_names
data_names_index_names = {name: i for i, name in enumerate(data_names_ordered)}

VV: int = 0
VH: int = 1
MASK: int = 2
BATHYMETRY: int = 3
WINDDIRECTION: int = 4
WINDQUALITY: int = 5
WINDSPEED: int = 6


category_labels = ["NonObject", "NonVessel", "NonFishingVessel", "FishingVessel"]
# classes for classification portion of model

BACKGROUND = 0
NONVESSEL = 1
NONFISHING = 2
FISHING = 3

N_CHANNELS = 7

S3BUCKET = ''
S3KEY_BASE = ''

S3ROOT = f's3://{S3BUCKET}/{S3KEY_BASE}'

cat_type = CategoricalDtype(categories=category_labels, ordered=True)
# Preset Basics
minimum = -50
color_scale = {
    'VV': (minimum, 20), 'VH': (minimum, 20), 'mask': (0,1), 'bathymetry': (-700,0), 'owiWindDirection': (0,360), 'owiWindQuality': (0,3), 'owiWindSpeed': (0,10),
}

# todo make WindDirection into a SinWindDirection and CosWindDirection to get NSEW

rescale_min_max = {
    'VV': (minimum, 35), 'VH': (minimum, 35), 'mask': (0,1), 'bathymetry': (-1000, 100), 'owiWindDirection': (0, 360), 'owiWindQuality': (0, 3), 'owiWindSpeed': (0,20),
}



in_s3 = 'SageMaker' in os.getcwd()