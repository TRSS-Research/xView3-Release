import numpy as np
# import jax.numpy as np
import os
import cv2
import pandas as pd
import time
from typing import Optional
import datetime as dt
from src import static
from src.object_detection import utils
import numba

root = os.getcwd()
today = str(dt.date.today()) + '_cfar'

logger = utils.get_logger(process='Process-Scene-CFAR', logger_name='CFAR')




def run_cfar(named_data: dict, configs: dict, diagnostic=False):
    """
        Implement the MR-CFAR algorithm with specifications from the configs file
    :param named_data: dictionary with named data (Keys are like VV, VH, mask, bathymetry...), values are loaded matrices
    :param configs: dictionary of configurations
    :return: Metadata, X, y
    """
    # setup
    start = time.time()

    for dataset in configs['resampling']:
        assert configs['resampling'][dataset] in static.resampling_dict

    # First get shapes and resize image and mask data
    orig_shape = named_data['VV'].shape
    cv_orig_dsize = orig_shape[1], orig_shape[0]

    cv_downsample_dsize = (int(orig_shape[1] * configs['sar_downsample_ratio']),
                           int(orig_shape[0] * configs['sar_downsample_ratio']))
    logger.debug('downsizing VV-SAR')
    vv_image_array_downsample = cv2.resize(named_data['VV'], dsize=cv_downsample_dsize,
                                           interpolation=static.resampling_dict[configs['resampling']['sar']])
    logger.debug('downsizing VH-SAR')
    vh_image_array_downsample = cv2.resize(named_data['VH'], dsize=cv_downsample_dsize,
                                           interpolation=static.resampling_dict[configs['resampling']['sar']])

    downsample_shape = vh_image_array_downsample.shape
    logger.debug(f'SAR downsample shape = {downsample_shape}')

    mask_array_downsample = cv2.resize(named_data['mask'], dsize=cv_downsample_dsize,
                                       interpolation=static.resampling_dict[configs['resampling']['mask']])

    logger.debug("Computing Look-at-mask")

    # Second get mask consisting of valid image data (i.e. not NODATA), and water data (ie. water mask)
    look_at_mask = utils.get_valid_water_mask(mask_array_downsample, vv_image_array_downsample)
    logger.info(f'Reviewing {look_at_mask.sum()} pixels')

    # Begin MultiResolution CFAR, first with downsampled data
    logger.info("Run CFAR on Downsampled VV")
    vv_flags_downsample = ca_cfar(
        mask_array=look_at_mask, minimum=static.minimum, image_array=vv_image_array_downsample,
        **configs['cfar']['downsample'])

    logger.info("Run CFAR on Downsampled VH")
    vh_flags_downsample = ca_cfar(
        mask_array=look_at_mask, minimum=static.minimum, image_array=vh_image_array_downsample,
        **configs['cfar']['downsample'])

    logger.info(f"downsample vv flag shape: = {vv_flags_downsample.shape}")
    logger.info(f"downsample vh flag shape: = {vh_flags_downsample.shape}")

    logger.info("Combining Downsample Flags")
    downsample_flags = np.logical_or(vv_flags_downsample, vh_flags_downsample).astype(np.uint8)

    # MRCFAR part 2: resize data and run CFAR on object detection mask
    down_up_flags = cv2.resize(downsample_flags, dsize=(orig_shape[1], orig_shape[0]), interpolation=0)
    logger.debug(f"downsample resized shape: {down_up_flags.shape}")

    ###
    logger.debug("Loading VV-SAR Med")
    cv_midsample_dsize = (int(configs['sar_midsample_ratio']*orig_shape[1]),
                          int(configs['sar_midsample_ratio']*orig_shape[0]))

    logger.debug('midsizing VV-SAR')
    vv_image_array_mid = cv2.resize(named_data['VV'], dsize=cv_midsample_dsize, interpolation=static.resampling_dict[configs['resampling']['sar']])
    logger.debug('midsizing VH-SAR')
    vh_image_array_mid = cv2.resize(named_data['VH'], dsize=cv_midsample_dsize, interpolation=static.resampling_dict[configs['resampling']['sar']])
    assert vh_image_array_mid.shape == vv_image_array_mid.shape

    midsample_shape = vh_image_array_mid.shape
    logger.info(f'SAR midsample shape = {midsample_shape}')

    logger.debug("Computing Look-at-mask")
    down_med_flags = cv2.resize(downsample_flags, dsize=cv_midsample_dsize, interpolation=0)

    logger.info(f'Reviewing {down_med_flags.sum()} pixels')
    logger.info("Run CFAR on midsampled VV")
    vv_flags_midsample = ca_cfar(
        mask_array=down_med_flags, minimum=static.minimum, image_array=vv_image_array_mid,
        **configs['cfar']['midsample'])

    logger.info("Run CFAR on midsampled VH")
    vh_flags_midsample = ca_cfar(
        mask_array=down_med_flags, minimum=static.minimum, image_array=vh_image_array_mid,
        **configs['cfar']['midsample'])

    midsample_flags = np.logical_or(vv_flags_midsample, vh_flags_midsample).astype(np.uint8)

    # MRCFAR part 3: resize data and run CFAR twice (width-wise and hieght-wise) on object detection mask
    mid_up_flags = cv2.resize(midsample_flags, dsize=(orig_shape[1], orig_shape[0]), interpolation=0)
    ###

    logger.info(f'Reviewing {mid_up_flags.sum()} pixels')

    v1_final_kwargs = configs['cfar']['full'].copy()
    v1_final_kwargs['guard_x'] += v1_final_kwargs['window_x'] - v1_final_kwargs['guard_x']

    logger.info('Run CFAR on Full VV v1')
    vv_flags_final_v1 = ca_cfar(mask_array=mid_up_flags, image_array=named_data['VV'], **v1_final_kwargs)

    logger.info('Run CFAR on Full VH v1')
    vh_flags_final_v1 = ca_cfar(mask_array=mid_up_flags, image_array=named_data['VH'], **v1_final_kwargs)

    v2_final_kwargs = configs['cfar']['full'].copy()
    v2_final_kwargs['guard_y'] += v2_final_kwargs['window_y'] - v2_final_kwargs['guard_y']

    logger.info('Run CFAR on Full VV v2')
    vv_flags_final_v2 = ca_cfar(mask_array=mid_up_flags, image_array=named_data['VV'], **v2_final_kwargs)

    logger.info('Run CFAR on Full VH v2')
    vh_flags_final_v2 = ca_cfar(mask_array=mid_up_flags, image_array=named_data['VH'], **v2_final_kwargs)

    combined_flags_final = np.logical_or.reduce((vv_flags_final_v1,vv_flags_final_v2,
                                                 vh_flags_final_v1,vh_flags_final_v2)).astype(np.uint8)


    # Identify Centroids
    logger.info('Computing Full Centroids')
    centroids, n_pixels = utils.get_flag_centroids(combined_flags_final, configs.get('n_pixel_threshold', 1))
    logger.info(f'{len(centroids)} contiguous centroids')

    # cut image by centroids to get stack of images
    logger.info("Exporting")
    X = utils.subset_image_arrays(centroids=centroids, named_data_dict=named_data, **configs['export_windows'])

    end = time.time()
    total_time = end-start
    logger.info(f'COMPLETED in: {total_time:.1f}')

    # If we are just runing the algorithm:
    if not diagnostic:
        return X, centroids,transform_n_pixels_to_length(n_pixels)
    # Otherwise, report intermeidate ata for performanc evaluation
    else:
        return X, centroids, transform_n_pixels_to_length(n_pixels), (downsample_flags, midsample_flags, combined_flags_final, cv_orig_dsize)


def transform_n_pixels_to_length(n_pixels: np.ndarray):
    return np.sqrt(n_pixels) * 10

def transform_length_to_n_pixels(length: np.ndarray):
    return (length / 10) ** 2 / 2


@numba.jit()
def ca_cfar(mask_array: utils.Image, image_array: utils.Image, window_x=6, window_y=6, guard_x=3, guard_y=3, threshold=2,
            minimum=-50):
    """
        Constant False Alarm Rate flagging algorithm
    :param mask_array: 2d array reflecting areas to review
    :param image_array: 2d array of images of interest
    :param window_x: x dimension to include in neighborhood
    :param window_y: y dimension to include in neighborhood
    :param guard_x: x dimension distance to center point
    :param guard_y: y dimension distance to center point
    :param threshold: threshold beyond which we raise a flag
    :param minimum: nodata flag
    :param thresholder: type of thresholder
    :return:
    """
    image_flags = np.zeros_like(image_array, dtype=np.bool_)
    # image_statistics = np.zeros_like(image_array)

    # zeros are both 'unknown' and 'water', so we need to find the intersection of
    #  'water/unknown' and 'water/land' to get just 'water'
    to_look_at_mask = np.where(mask_array)
    to_look_at_stack = np.stack(to_look_at_mask, axis=1)

    # need to get a unraveled mask to hide the 'guard' cells.
    keep_array = np.ones((window_y * 2, window_x * 2), dtype=np.bool_).flatten()
    drop_ranges = [
        (i * window_y * 2 + window_y - guard_y, i * window_y * 2 + window_y + guard_y)
        for i in range(window_x - guard_x, 2 * window_x - guard_x)
    ]
    for start, end in drop_ranges:
        keep_array[start:end] = False

    # iterate over all nonzeros
    for i, j in to_look_at_stack:
        flattened = image_array[i - window_y: i + window_y,
                                j - window_x: j + window_x].flatten()

        flattened = flattened[keep_array[:flattened.shape[0]] & (flattened > minimum)]

        if len(np.unique(flattened)) < 2:
            image_flags[i, j] = False
        else:
            std = flattened.std()
            if std > 0:
                statistic = (image_array[i, j] - flattened.mean()) / std
                image_flags[i, j] = statistic > threshold
            else:
                image_flags[i, j] = False
    return image_flags


def process_scene_cfar(scene_id: str, jobname: str, data_folder: str, df_scene: pd.DataFrame,
            configs_name: str = 'base_configs.yml',) -> (static.MetadataDataFrame, static.TrainData, static.TestData):
    """
        Load a scene and run MRCFAR to get a stack of predictions. This runs Diagnostics on CFAR so it will be slower,
        and will report positive and negative results accordingly, and *requires* the standard metadata file
    :param scene_id: scene-id (expected to be folder name)
    :param jobname: training/testing/validation
    :param data_folder: base location of data folder
    :param df_scene: DataFrame with positive labels
    :param configs_name: filename for configs.yaml file
    :return: Tuple of metadata, images, and prediction values
    """

    # Load Configs and Data
    configs = utils.load_configs(configs_path=os.path.join('configs', 'cfar', configs_name))
    named_data = utils.load_image_data_from_folder(data_folder, configs['resampling'])

    # RUN MRCFAR algorithm
    X, centroids, length, diagnostics = run_cfar(named_data = named_data, configs=configs, diagnostic=True)

    downsample_flags, midsample_flags, combined_flags_final, cv_orig_dsize = diagnostics

    label_row_cols = df_scene[static.metadata_centroid_columns].values

    # Begin to analyze performance at different resolutions
    ## Evaluate Downsample Performance
    logger.info('Getting Downsample Centroids')
    centroid_flags_downsample = np.zeros_like(downsample_flags, dtype=np.uint8)
    centroids_downsample, _ = utils.get_flag_centroids(downsample_flags)
    logger.info(f'{len(centroids)} contiguous centroids')

    ### Create Downsample Flag Image
    for i,j in centroids_downsample:
        centroid_flags_downsample[i,j] = 1
    centroid_flags_downsample_up = cv2.resize(centroid_flags_downsample, dsize=cv_orig_dsize, interpolation=0)

    ### Evaluate downsample Flags in performance on metatdata
    logger.info('Checking downsample Performance')
    downsample_performance = utils.check_performance(
        label_row_cols, centroid_flags_downsample_up, downsample_factor=configs['sar_downsample_ratio'],
        **configs['check-performance']['downsample'])

    ## Midsample Performance
    logger.info('Getting midsample Centroids')
    centroid_flags_midsample = np.zeros_like(midsample_flags, dtype=np.uint8)
    centroids_midsample, _ = utils.get_flag_centroids(midsample_flags)

    for i, j in centroids_midsample:
        centroid_flags_midsample[i,j] = 1
    centroid_flags_midsample_up = cv2.resize(centroid_flags_midsample, dsize=cv_orig_dsize, interpolation=0)

    logger.info('Checking Midsample Performance')
    midsample_performance = utils.check_performance(
        label_row_cols, centroid_flags_midsample_up, downsample_factor=configs['sar_midsample_ratio'],
        **configs['check-performance']['midsample'])

    ## Full Resolution Performance
    centroid_flags = np.zeros_like(combined_flags_final)
    for i, j in centroids:
        centroid_flags[i,j] = 1

    logger.info('Checking Final Performance')
    full_performance = utils.check_performance(
        label_row_cols, centroid_flags, downsample_factor=1,
        **configs['check-performance']['full'])

    df_performance = pd.DataFrame([downsample_performance, midsample_performance, full_performance], columns=['TP', 'FP','FN','TN'])
    df_performance.loc[:, 'precision'] = df_performance['TP'] / (df_performance['TP'] + df_performance['FP'])
    df_performance.loc[:, 'recall'] = df_performance['TP'] / (df_performance['TP'] + df_performance['FN'])
    df_performance.loc[:, 'F1'] = 2 * df_performance['recall'] * df_performance['precision'] / (df_performance['recall'] + df_performance['precision'])
    logger.info(f'\n{df_performance.head()}\n')
    df_performance.loc[:, 'downsample-ratio'] = [configs['sar_downsample_ratio'],configs['sar_midsample_ratio'], 1]
    df_performance.loc[:, 'scene_id'] = scene_id
    for cfar_type in configs['cfar']:
        for key in configs['cfar'][cfar_type]:
            df_performance.loc[:, f'cfar-{cfar_type}-{key}'] = configs['cfar'][cfar_type][key]

    output_path = static.performance_output_path
    if os.path.exists(output_path):
        df_performance.to_csv(output_path, mode='a', index=False, header=False)
    else:
        df_performance.to_csv(output_path, index=False)
    logger.info('Completed')

    centroids = centroids[np.lexsort(centroids.T, axis=0)] # sort them
    df_mapped = utils.map_single_scene_centroids_to_metadata(df_scene, predict_centroids=centroids, scene_id=scene_id, threshold=20, include_missed_positives=False)
    if df_mapped.shape[0] != centroids.shape[0]:
        logger.warning("Shapes Differ")
    elif not (centroids == df_mapped[['prediction-row', 'prediction-col']].values).all():
        logger.warning("Centroid - true-positive mapped are not the same")

    df_mapped.loc[:, 'label'] = df_mapped.apply(utils.get_label_apply, axis=1)
    y = df_mapped['label'].astype(static.cat_type).cat.codes.values

    # save the images
    if configs.get('save_intermediate', False):
        logger.info(f'Saving Files')
        data_output_folder = os.path.join(root, 'cfar-output', jobname)
        utils.save_intermediate_separate(data_output_folder, scene_id, df_mapped, X, y)

    return df_mapped, (X, y)


def cfar_inference(data_folder: str, configs_name: str='base_configs.yml') -> (static.TrainData, static.Centroids):
    """
        Prediction time MRCFAR, no diagnostics
    :param data_folder: direct folder of image data
    :param configs_name: configs filen ame
    :return:
    """
    configs = utils.load_configs(configs_path=os.path.join('configs', 'cfar', configs_name))
    logger.info(f'Loading from {data_folder}')
    named_data = utils.load_image_data_from_folder(data_folder, configs['resampling'])
    X, centroids, length = run_cfar(named_data=named_data, configs=configs, diagnostic=False)
    return X, centroids, length


def process_scene_cfar_handler(scene_id: str, jobname:str, folder_name: str, s3=False, configs_name='base_configs.yml', df_metadata: Optional[pd.DataFrame] = None):
    if s3:
        data_folder = f'{static.S3ROOT}/{folder_name}/{scene_id}'
    else:
        data_folder = f'data/{folder_name}/{scene_id}'
    if df_metadata is None:
        if folder_name not in ['training','validation']:
            raise NotImplementedError
        df_metadata = utils.read_csv(utils.get_metadata_paths()[folder_name])

    return process_scene_cfar(scene_id, jobname=jobname, data_folder=data_folder,
                              df_scene=df_metadata.loc[df_metadata['scene_id'] == scene_id], configs_name=configs_name)


if __name__ == '__main__':
    # scene_id ='0222c037f32357b7t'
    scene_id = '00a035722196ee86t'
    jobname = 'test1-cfar-train'
