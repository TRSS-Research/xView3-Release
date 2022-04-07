import functools
import pandas as pd
import rasterio
import numpy as np
import tqdm
from random import shuffle
from typing import Tuple, Optional, List, Dict
import matplotlib.pyplot as plt
import logging
import os
import scipy.stats
import scipy.signal
import yaml
from src import static
import boto3
import datetime as dt
import numba
from scipy.spatial import KDTree
today = str(dt.date.today())

Image = np.ndarray
Coords = Tuple[int, int]
FlaggedPixelArray = np.ndarray
SceneCentroidDict = Dict[str, np.ndarray]


def map_single_scene_centroids_to_metadata(df_scene, predict_centroids, scene_id, threshold=10, include_missed_positives=True):
    true_centroids = df_scene[static.metadata_centroid_columns].values

    guess_tree = KDTree(predict_centroids)
    true_to_pred_dists, true_to_pred_indexes = guess_tree.query(true_centroids)

    df_scene.loc[:,'dists'] = true_to_pred_dists
    df_scene.loc[:,'prediction-row'] = predict_centroids[true_to_pred_indexes][:, 0]
    df_scene.loc[:,'prediction-col'] = predict_centroids[true_to_pred_indexes][:, 1]
    df_scene.loc[:,'found'] = true_to_pred_dists < threshold

    predict_centroids_which_hit = true_to_pred_indexes[true_to_pred_dists < threshold]

    predict_centroids_which_miss = np.ones(len(predict_centroids), np.bool)
    predict_centroids_which_miss[predict_centroids_which_hit] = 0

    df_false_positives = pd.DataFrame(predict_centroids[predict_centroids_which_miss], columns=['prediction-row', 'prediction-col'])

    try:
        df_false_positives.loc[:, 'scene_id'] = scene_id
    except ValueError as noissues:
        pass
    if not include_missed_positives:
        df_scene = df_scene.loc[df_scene['found'], :]
    df_scene = df_scene.append(df_false_positives)
    df_scene = df_scene.sort_values(['prediction-col','prediction-row'])

    return df_scene



def get_label_apply(x):
    if pd.notnull(x['is_fishing']):
        if x['is_fishing']:
            return 'FishingVessel'
        else:
            return 'NonFishingVessel'
    elif pd.notnull(x['is_vessel']):
        if x['is_vessel']:
            return 'NonFishingVessel'
        else:
            return 'NonVessel'
    else:
        return "NonObject"


def map_centroids_to_metadata(scene_centroid_dict: dict, training_or_validation='training', threshold=10):
    fn = 'train' if training_or_validation =='training' else training_or_validation
    df = read_csv(f'data/{training_or_validation}/{fn}.csv')
    df_false_positives_list = []
    for scene_id in scene_centroid_dict:
        predict_centroids = scene_centroid_dict[scene_id]
        true_centroids_index = df.loc[df['scene_id'] == scene_id].index
        true_centroids = df.loc[true_centroids_index, static.metadata_centroid_columns].values

        guess_tree = KDTree(predict_centroids)
        true_to_pred_dists, true_to_pred_indexes = guess_tree.query(true_centroids)
        df.loc[true_centroids_index, 'dists'] = true_to_pred_dists
        df.loc[true_centroids_index, 'prediction-row'] = predict_centroids[true_to_pred_indexes][:, 0]
        df.loc[true_centroids_index, 'prediction-col'] = predict_centroids[true_to_pred_indexes][:, 1]
        df.loc[true_centroids_index, 'found'] = true_to_pred_dists < threshold

        predict_centroids_which_hit = true_to_pred_indexes[true_to_pred_dists < threshold]

        predict_centroids_which_miss = np.ones(len(predict_centroids), np.bool)
        predict_centroids_which_miss[predict_centroids_which_hit] = 0

        df_false_positives = pd.DataFrame(predict_centroids_which_miss, columns=['prediction-row','prediction-col'])
        df_false_positives['scene-id'] = scene_id
        df_false_positives_list.append(df_false_positives)

    df_false_positives = pd.concat(df_false_positives_list, axis=0)

    return df, df_false_positives


def load_image_data_from_paths_dict(paths_dict, resamplers):
    try:
        orig_image_array_vv = load_sar(paths_dict['VV'], minimum=static.minimum)
        orig_image_array_vh = load_sar(paths_dict['VH'], minimum=static.minimum)
        assert orig_image_array_vh.shape == orig_image_array_vv.shape
        named_data = {name: load_supplemental(
            path=paths_dict[name], resampler=static.rasterio_resampling_dict[resamplers[name]], shape=orig_image_array_vh.shape)
            for name in static.supplemental_data_names
        }
    except (rasterio._err.CPLE_AppDefinedError, rasterio.errors.RasterioIOError):
        raise AWSCorruptedError('Cannot Read File')
    named_data['VH'] = orig_image_array_vh
    named_data['VV'] = orig_image_array_vv
    return named_data


def load_image_data_from_folder(folder, resamplers):
    paths_dict = get_image_paths(folder)
    return load_image_data_from_paths_dict(paths_dict=paths_dict,resamplers=resamplers)


class AWSCorruptedError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


def load_data_numpy(paths_dict: Dict, resamplers: Dict):
    print('VV')
    orig_image_array_vv = load_sar(paths_dict['VV'], minimum=static.minimum)
    print('VH')
    orig_image_array_vh = load_sar(paths_dict['VH'], minimum=static.minimum)
    assert orig_image_array_vh.shape == orig_image_array_vv.shape

    # output_array = np.zeros((len(static.data_names_ordered), *orig_image_array_vh.shape), dtype=np.float32)
    print('Supplementals')
    supplementals = tuple(load_supplemental(
        path=paths_dict[name], resampler=static.rasterio_resampling_dict[resamplers[name]], shape=orig_image_array_vh.shape)
                          for name in static.supplemental_data_names
    )
    print('Combining')
    return combine_arrays((orig_image_array_vv , orig_image_array_vh) + supplementals)


@numba.jit()
def combine_arrays(arrays):
    return np.stack(arrays)


def load_data_and_configs(root, training_or_validation, scene_id, configs_path):
    paths_dict = get_paths(root, training_or_validation, scene_id)
    configs = load_configs(configs_path)
    named_data = load_image_data_from_paths_dict(paths_dict, configs['resampling'])
    return named_data, configs


def load_configs(configs_path):
    with open(configs_path, 'r') as configs_file:
        configs = yaml.load(configs_file, Loader=yaml.FullLoader)
    return configs


def get_paths(root, training_or_validation, scene_id):
    return {
        'training': os.path.join(root, 'data', 'train.csv'),
        'validation': os.path.join(root, 'data', 'validation.csv'),
        'bathymetry': os.path.join(root, 'data', training_or_validation, scene_id, 'bathymetry.tif'),
        'owiWindDirection': os.path.join(root, 'data', training_or_validation, scene_id, 'owiWindDirection.tif'),
        'owiWindQuality': os.path.join(root, 'data', training_or_validation, scene_id, 'owiWindQuality.tif'),
        'owiWindSpeed': os.path.join(root, 'data', training_or_validation, scene_id, 'owiWindSpeed.tif'),
        'VV': os.path.join(root, 'data', training_or_validation, scene_id, 'VV_dB.tif'),
        'VH': os.path.join(root, 'data', training_or_validation, scene_id, 'VH_dB.tif'),
        'mask': os.path.join(root, 'data', training_or_validation, scene_id, 'owiMask.tif'),
    }


def get_metadata_paths():
    return {'training': 'data/train.csv', 'validation': 'data/validation.csv'}


def get_paths_s3(root, training_or_validation, scene_id):
    return {
        'training': os.path.join(root, 'data', 'train.csv'),
        'validation': os.path.join(root, 'data', 'validation.csv'),
        'bathymetry': f'{static.S3ROOT}/{training_or_validation}/{scene_id}/bathymetry.tif',
        'owiWindDirection': f'{static.S3ROOT}/{training_or_validation}/{scene_id}/owiWindDirection.tif',
        'owiWindQuality': f'{static.S3ROOT}/{training_or_validation}/{scene_id}/owiWindQuality.tif',
        'owiWindSpeed': f'{static.S3ROOT}/{training_or_validation}/{scene_id}/owiWindSpeed.tif',
        'VV': f'{static.S3ROOT}/{training_or_validation}/{scene_id}/VV_dB.tif',
        'VH': f'{static.S3ROOT}/{training_or_validation}/{scene_id}/VH_dB.tif',
        'mask': f'{static.S3ROOT}/{training_or_validation}/{scene_id}/owiMask.tif',
    }


def get_image_paths(folder):
    return {'bathymetry': os.path.join(folder, 'bathymetry.tif'),
            'owiWindDirection': os.path.join(folder, 'owiWindDirection.tif'),
            'owiWindQuality': os.path.join(folder, 'owiWindQuality.tif'),
            'owiWindSpeed': os.path.join(folder,'owiWindSpeed.tif'),
            'VV': os.path.join(folder,'VV_dB.tif'),
            'VH': os.path.join(folder,'VH_dB.tif'),
            'mask': os.path.join(folder,'owiMask.tif'),
            }

if static.in_s3:
    get_paths = get_paths_s3


def get_logger(process, logger_name):
    log = logging.getLogger(logger_name)
    if len(log.handlers) > 0:
        return log
    log.setLevel(level=logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler for logger.
    fh = logging.FileHandler(f'logs/{process}.log')
    fh.setLevel(level=logging.DEBUG)
    fh.setFormatter(formatter)

    # create console handler for logger.
    ch = logging.StreamHandler()
    ch.setLevel(level=logging.INFO)
    ch.setFormatter(formatter)

    # add handlers to logger.
    log.addHandler(fh)
    log.addHandler(ch)
    return log


def load_sar(path: str, minimum: int, resample_factor: Optional[float] = None, resampling=rasterio.enums.Resampling.bilinear, convolve=None):
    """
        load the SAR data
    :param path: str
    :param minimum: below this we censor the 'nodata' character from rasterio
    :param resample_factor: downsample ratio
    :param resampling: how to infer between pixels
    :return:
    """
    with rasterio.open(path) as src:
        if resample_factor is not None:  # if we are resampling / making the image smaller
            image_array = src.read(
                out_shape=(
                    src.count,
                    int(src.height * resample_factor),
                    int(src.width * resample_factor)
                ),
                resampling=resampling
            ).squeeze()
        else:
            image_array = src.read(1)
    # nan seems to be reported as a large negative number, so we need to mask this to faciliate plotting
    image_array[image_array < minimum] = minimum
    if convolve is not None:
        scipy.signal.convolve2d(image_array, convolve, mode='valid')
    return image_array



def load_supplemental(path:str, shape: Tuple[int], resampler=rasterio.enums.Resampling.bilinear)-> np.ndarray:
    """
        Need to make all the inputs the same resolution as the base image
    :param path: input_path to read
    :param shape: dimensionality of the base
    :param resampler: how to infer cross-pixel correlation
    :return: image_array
    """
    with rasterio.open(path) as src:
        image_array = src.read(out_shape=shape, resampling=resampler).squeeze()
    return image_array


def check_correct_window(row: int, col: int, flag_array: np.ndarray,
                         correction_window_x: int = 0, correction_window_y: int = 0):
    sub_image = zoom_in_on_pixel(flag_array, row, col, window_x=max(1, correction_window_x),
                                                       window_y=max(1, correction_window_y))

    return check_correct(sub_image)


def check_correct(windowed_array):
    return (windowed_array > 0).any()


def check_performance(row_cols: np.ndarray, flag_array: Image, correction_window_x: int = 0,
                      correction_window_y: int = 0, downsample_factor: float = 1) -> List[int]:
    """
        Report Performance on Window
    :param row_cols: iterable where each iteration returns the row and column as i,j
    :param flag_array:
    :param logger: logging function
    :param correction_window_x:
    :param correction_window_y:
    :param downsample_factor:
    :return:
    """
    n = 0
    for i, j in row_cols:
        n += check_correct_window(i, j, flag_array, correction_window_x, correction_window_y)
    tp = n
    fp = int(flag_array.sum()*downsample_factor**2 - n)
    fn = int(row_cols.shape[0] - n)
    tn = int(flag_array.size * downsample_factor**2 - row_cols.shape[0])
    return [tp, fp, fn, tn]


def load_subset_data(filepath, scene_id):
    df = read_csv(filepath)
    return df.loc[df['scene_id'] == scene_id,:]


@functools.lru_cache(None)
def read_csv(filepath):
    # cacheing this so we dont have to reload it a lot
    return pd.read_csv(filepath)


# Plotting and pixel associations

def get_flag_centroids(flag_matrix: Image, pixel_filter_threshold=1) -> (FlaggedPixelArray, np.ndarray):
        # todo use weighted flag matrix?
        # todo output centroid matrix?
        # todo combine these together?
    """
        Get centerpoints of contiguous flags
    :param flag_matrix: matrix of pixel-wise flags
    :param logger: print/logging function
    :return: center-of-mass of flag_matrix
    """
    _, groups = get_pixel_groups(flag_matrix)

    centroids, n_pixels = map(np.array,
                              zip(*((np.mean(groups[root_coord], axis=0, dtype=np.int), len(groups[root_coord]))
                                    for root_coord in groups)))

    pixel_array = np.array(n_pixels)
    pixel_filter = pixel_array >= pixel_filter_threshold

    return centroids[pixel_filter], pixel_array[pixel_filter]

def zoom_in_on_pixel(image_array: Image, row: int, col: int, window_x: int=25, window_y: int=25) -> Image:
    """
        subset image_array to be centered on (row, col)
        # todo figure out how to handle the edges
    """
    return image_array[row - window_y:row + window_y, col - window_x:col + window_x]


def plot_images(row_cols: FlaggedPixelArray, named_data_dict, centroid_labels: List[str],
                    alt_centers: Optional[FlaggedPixelArray] = None, window_x: int = 32, window_y: int = 32,
                    root_out: str = 'output/images', suffix: str = 'centered'):
    """
        Plot and Save images.
    :param row_cols: iterable where each iteration yields the row,col reflecting a pixel to center an image upon
    :param name_list: list of names for image titles, aligned with image_list
    :param image_list: list of image_arrays for plotting, aligned with name_list
    :param alt_centers: alternative iterable of coordinates which should be plotted on same axes. m
                         must have same shape as `row_cols`
    :param window_x: width below and above center pixel to plot output image
    :param window_y: height below and above center pixel to plot output image
    :param root_out: where to save images
    :param suffix: filename parameter for
    :return: None
    """
    if alt_centers is not None:
        assert alt_centers.shape == row_cols.shape
    assert len(centroid_labels) == len(row_cols)
    for centroid_ix, (i, j) in enumerate(row_cols):
        f, axarr = plt.subplots(1, len(named_data_dict), figsize=(3 * len(named_data_dict), 3))
        for ix, name in enumerate(static.data_names_ordered):
            image_array = named_data_dict[name]
            axarr[ix].imshow(zoom_in_on_pixel(image_array, row=i, col=j, window_x=window_x, window_y=window_y))
            axarr[ix].set_title(f'{name}-{centroid_labels[centroid_ix]}',
                                fontdict={'fontsize':15 - len(named_data_dict)})
            axarr[ix].plot(window_y, window_x, 'w+', markersize=4)
            if alt_centers is not None:
                axarr[ix].plot(alt_centers[centroid_ix][0] - i + window_y,
                               alt_centers[centroid_ix][1] - j + window_x,
                               'r+', markersize=4)
        f.savefig(os.path.join(root_out, f'{suffix}_{centroid_ix}.png'))


def lookup_base_pixel(lookup_dict: dict, coords: Coords) -> Coords:
    """
        follow trail in a dict to get to the final pixel
    :param lookup_dict:
    :param coords:
    :return:
    """
    if coords not in lookup_dict:
        print(f'Cannot Find {coords}')
        return coords
    while coords != lookup_dict[coords]:
        coords = lookup_dict[coords]
    return coords


def get_pixel_groups(flag_matrix: Image) -> Tuple[dict, dict]:
    """
        conglomerate pixel-level flags into flag groups
    :param flag_matrix: 2d matrix with 1s reflecting flagged pixels
    :param logger: logging function
    :return: (lookup_dict, group_dict)
    """
    row_cols = np.stack(np.where(flag_matrix > 0)).T
    lookup_dict = dict()
    groups = dict()
    for i, j in tqdm.tqdm(row_cols):
        to_check = [(i - 1, j),
                    (i, j - 1),
                    (i - 1, j - 1),
                    (i - 1, j + 1)]
        for check in to_check:
            if check in lookup_dict:
                lookup_dict[(i, j)] = check

        if (i, j) not in lookup_dict:
            lookup_dict[(i, j)] = (i, j)
            groups[(i, j)] = [(i, j)]
    for base_coords in tqdm.tqdm(lookup_dict):
        coords = base_coords
        if base_coords in groups:
            continue
        else:
            groups[lookup_base_pixel(lookup_dict, coords)].append(base_coords)
    return lookup_dict, groups


# CA CFAR implementation

def z_score_thresholder(cell: float, sample: np.ndarray, threshold: float) -> Tuple[float, bool]:
    statistic = (cell - sample.mean()) / sample.std()
    return statistic, statistic > threshold


def t_thresholder(cell: float, sample: np.ndarray, threshold: float) -> Tuple[float, bool]:
    args = scipy.stats.t.fit(sample)
    cdf = scipy.stats.t.cdf(cell, *args)
    return cdf, cdf > threshold


def gamma_thresholder(cell: float, sample: np.ndarray, threshold: float) -> Tuple[float, bool]:
    args = scipy.stats.gamma.fit(sample)
    cdf = scipy.stats.gamma.cdf(cell, *args)
    return cdf, cdf > threshold


def basic_thresholder(cell: float, sample: np.ndarray, threshold: float) -> Tuple[float, bool]:
    statistic = cell / sample.mean()
    return statistic, statistic > threshold


def norm_data(x):
    return (x - x.min()) / (x.max() - x.min())


def save_npy(path, array):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)


def subset_image_arrays(centroids: FlaggedPixelArray, named_data_dict: dict, window_x: int = 32, window_y: int = 32):
    """
        Slice and stack images
    :param centroids: 2 column array with X and Y components
    :param named_data_dict: the data with keys as data names, and images as values
    :param window_x: x dimension window around center-point
    :param window_y: y dimension window around center-point
    :return: stack of images
    """
    output_array = np.zeros((len(centroids), 2 * window_y, 2 * window_x, len(static.data_names_ordered)))
    for centroid_index, centroid in tqdm.tqdm(enumerate(centroids), total=len(centroids)):
        for output_array_index, name in enumerate(static.data_names_ordered):
            output_array[centroid_index, :, :, output_array_index] = zoom_in_on_pixel(
                named_data_dict[name], row=centroid[0], col=centroid[1], window_x=window_x, window_y=window_y
            )
    return output_array





def subset_image_arrays_npy(centroids: FlaggedPixelArray, ordered_image_stack: np.ndarray,
                            window_x: int=32, window_y: int=32):
    output_array = np.zeros((len(centroids), len(static.data_names_ordered), 2 * window_y, 2 * window_x), dtype=np.float64)
    for centroid_ix, centroid in enumerate(centroids):
        output_array[centroid_ix, ...] = ordered_image_stack[:, centroid[0] - window_y: centroid[0] + window_y,
                                                                centroid[1] - window_x: centroid[1] + window_x]
    return output_array


def save_intermediate(folder, scene_id, df, X, y):
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, scene_id + '_X.npy'), X)
    np.save(os.path.join(folder, scene_id + '_y.npy'), y)
    df.to_csv(os.path.join(folder, f'{scene_id}_metadata.csv'))


def save_intermediate_separate(folder:str, scene_id:str, df: pd.DataFrame, X: np.ndarray, y: np.ndarray):
    """
        Save stack of images to given folder
    :param folder: directory within scene_id is apparent
    :param scene_id:
    :param df: metadata dataframe
    :param X: stack of multi-channel images
    :param y: stack of predictions
    :return: None
    """
    folder = os.path.join(folder, scene_id)
    os.makedirs(folder, exist_ok=True)
    if df.shape[0] != X.shape[0] or X.shape[0] != y.shape[0]:
        raise Exception(f'Data sizes are off... X: {X.shape[0]}, y: {y.shape[0]}, df: {df.shape[0]}')

    for i in range(X.shape[0]):
        np.save(os.path.join(folder, f'X_{i}.npy'), X[i, ...])
        np.save(os.path.join(folder,  f'y_{i}.npy'), y[i])
        df.iloc[i, :].to_csv(os.path.join(folder, f'metadata_{i}.csv'), header=True)


@numba.jit(nopython=True)
def get_valid_water_mask(mask_array, polarized_array):
    return np.multiply((mask_array < .9)
                       ,(polarized_array > static.minimum+.1))


def get_scene_list_folder(folder, limit=(0,20), random=False):
    scene_list = sorted(os.listdir(folder))
    if random:
        shuffle(scene_list)
    return scene_list[limit[0]:limit[1]]


def get_scene_list_s3(data_folder_name, limit=(0, 20), random=False):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=static.S3BUCKET, Prefix=f'{static.S3KEY_BASE}/{data_folder_name}/', Delimiter='/')
    objects = [i['Prefix'].split('/')[-2] for i in response.get('CommonPrefixes',[])]

    scene_list = sorted(objects)
    if random:
        shuffle(scene_list)
    return scene_list[limit[0]:limit[1]]


def load_to_s3(name, data_folder_name):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(static.S3BUCKET)
    s3_output_root = 'shared/xview3/data/segmented'

    local_data_folder = f'output/{name}/{data_folder_name}'
    for f in ['metadata.csv','X.npy','y.npy']:
        bucket.upload_file(os.path.join(local_data_folder, f),
                           f'{s3_output_root}/{f}')

def load_to_s3_generic(load_path_local, s3_key_root, bucket='data-science-sagemaker-bucket'):
    if not os.path.exists(load_path_local):
        raise Exception(f'Cannot Find: {load_path_local}')
    io_paths = []
    if os.path.isdir(load_path_local):
        print(f'Searching for files under {load_path_local}')
        for root, dirs, files in os.walk(load_path_local):
            temproot = root.replace(load_path_local+'/','')
            for f in files:
                io_paths.append((os.path.join(root, f), os.path.join(s3_key_root, temproot, f)))
    elif os.path.isfile(load_path_local):
        fn = os.path.basename(load_path_local)
        io_paths.append((load_path_local, os.path.join(s3_key_root, fn)))
    else:
        raise Exception(f"Not sure what {load_path_local} is")
    s3 = boto3.resource('s3')

    bucket = s3.Bucket(bucket)
    for input_path, output_path in io_paths:
        print(input_path, output_path)
        bucket.upload_file(input_path, output_path)

        