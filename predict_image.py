import threading

from src.object_detection.object_detection_cfar import cfar_inference
from src.object_detection.utils import get_logger, AWSCorruptedError
from src.classifier import nn_classifier, data_loader
import time
name = str(int(time.time()))

import random
import os
import pandas as pd
import numpy as np
import boto3
from src import static
from src.object_detection.utils import save_npy


logger = get_logger('predict-image','predict-image')

def get_scenelist(folder):
    if folder.startswith('s3'):
        s3 = boto3.resource('s3')
        path_pieces = folder.split('//')[1].split('/')
        bucket = s3.Bucket(path_pieces[0])
        scene_ids = [object_summary.key.split('/')[-2] 
                     for object_summary in bucket.objects.filter(Prefix='/'.join(path_pieces[1:]))]
        scene_ids = list(set(scene_ids))
    else:
        scene_ids = [scene_id for scene_id in os.listdir(folder) 
                 if os.path.isdir(os.path.join(folder, scene_id))]
    return sorted(scene_ids)


def main(args):
    start = time.time()
    data_root = args.image_folder

    if args.scene_ids is not None:
        scene_ids = args.scene_ids.split(",")
    else:
        scene_ids = get_scenelist(data_root)
    if args.invert_scene_list:
        scene_ids = scene_ids[::-1]
    if args.shuffle_scene_list:
        random.shuffle(scene_ids)
    class_channels_ix = tuple(args.class_channel_indexes)
    length_channels_ix = tuple(args.length_channel_indexes)
    
    cfar_configs_name = args.cfar_configs_name

    # Create output directories if it does not already exist
    fp = args.output
    output_folder  = os.path.dirname(fp)
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
       
    logger.info(f'Loading Model:{args.class_model_path}')
    class_model = nn_classifier.load_model(args.class_model_path)
    
    logger.info(f'Loading Model:{args.length_model_path}')
    length_model = nn_classifier.load_model(args.length_model_path)
    data = []
    n = len(scene_ids)
    
    for scene_no, scene_id in enumerate(scene_ids):
        scene_start = time.time()
        logger.info(f'{scene_id} -- {scene_no} of {n}.')
        
        cfar_output_folder = os.path.join(args.cfar_output_root, scene_id)
        centroids_filepath = os.path.join(cfar_output_folder, 'centroids.npy')
        X_filepath = os.path.join(cfar_output_folder, 'X.npy')
        length_filepath = os.path.join(cfar_output_folder, 'length.npy')
        if not (os.path.exists(centroids_filepath) and os.path.exists(X_filepath)) or\
                args.overwrite:
            logger.info(f'Running CFAR.')
            try:
                X, centroids, length = cfar_inference(data_folder=os.path.join(data_root, scene_id), configs_name=cfar_configs_name)
            except AWSCorruptedError as e:
                logger.warning(f'\t\tSkipping {scene_id}, corrupted\t\t {e}')
                continue
            if args.save_cfar:
                save_npy(centroids_filepath, centroids)
                save_npy(X_filepath, X)
                save_npy(length_filepath, length)
        else:
            logger.info(f'Loading CFAR from {cfar_output_folder}.')
            X, centroids, length = np.load(X_filepath), np.load(centroids_filepath), np.load(length_filepath)
        
        X_class = rescale_X(X, class_channels_ix)
        X_length = rescale_X(X, length_channels_ix)
       
        logger.info(f'Running Predict on {X.shape[0]} observations')
        y_pred = class_model.predict(X_class)
        y_pred = np.argmax(y_pred, axis=1)
        
        logger.info(f'Estimating Length.')
        length_preds = length_model.predict(X_length)[:, 0]
        
        for ix in range(centroids.shape[0]):
            prediction_val = y_pred[ix]
            if prediction_val != 0:
                row, col = centroids[ix] 
                is_vessel = prediction_val > 1
                is_fishing_vessel = prediction_val == 3
                data.append(
                    [row, col, scene_id, is_vessel, is_fishing_vessel, length_preds[ix]]
                )

        logger.info(f'\t{(y_pred == 0).sum()} NonObjects (False Positives from CFAR)')
        logger.info(f'\t{(y_pred == 1).sum()} Objects')
        logger.info(f'\t{(y_pred == 2).sum()} NonFishingVessels')
        logger.info(f'\t{(y_pred == 3).sum()} FishingVessels')
        logger.info(f'Scene Time Taken: {time.time() - scene_start:.2f}')
        
    df_out = pd.DataFrame(data=data,
        columns=(
            "detect_scene_row",
            "detect_scene_column",
            "scene_id",
            "is_vessel",
            "is_fishing",
            "vessel_length_m",
        )
    )
    df_out.to_csv(fp +'_excess.csv', index=False)
    print(f"{len(df_out)} detections found")

    df_out = reduce_nearby_predictions(df_out, threshold=10)
    df_out.to_csv(fp, index=False)
    print(f"{len(df_out)} detections found")


def rescale_X(X, ix):
    rescalers = data_loader.get_rescalers(X[0, ..., 0].shape)
    multiply = rescalers[0][..., ix]
    add = rescalers[1][..., ix]
    return X[..., ix] * multiply[np.newaxis,...] + add[np.newaxis,...]


NonFishingVessel_avg_length = 110
FishingVessel_avg_length = 30


def posthoc_adjust_length(lengths, preds):
    expected = np.ones_like(lengths)
    expected[preds == static.NONFISHING] = NonFishingVessel_avg_length
    expected[preds == static.FISHING] = FishingVessel_avg_length
    expected[expected==1] = 2 * lengths[expected==1]  # fill in the rest with themselves
    return length_mediator(lengths, expected)


def length_mediator(actual, expected):
        return (actual + expected) / 2


def get_keep_indices(df_scene, threshold=10):
    centroids = df_scene.loc[:, ["detect_scene_row","detect_scene_column"]].values
    too_closes = list()
    separates = list()
    for ix, coord in enumerate(centroids):
        too_close = np.where((abs(centroids - coord) < np.array((threshold,threshold))).all(axis=1))[0]

        if len(too_close) > 1:
            too_closes.append((ix, *tuple(too_close)))
        else:
            separates.append(ix)
    too_closes_sets = [tuple(set(group)) for group in too_closes]
    too_close_groups = list(set(too_closes_sets))

    too_closes_choices = []
    for close_pair in too_close_groups:
        df_pair = df_scene.iloc[np.array(close_pair), :]
        too_closes_choices.append(
            close_pair[np.argmax(df_pair['vessel_length_m'].values)]
        )
    indexes = list(set(separates + too_closes_choices))
    return indexes


def reduce_nearby_predictions(df_pred, threshold=8):
    dfs_out = []
    for scene in df_pred['scene_id'].unique():
        df_scene = df_pred.loc[df_pred['scene_id']==scene,:]
        indexes = get_keep_indices(df_scene, threshold)
        dfs_out.append(df_scene.iloc[indexes,:])
    return pd.concat(dfs_out)




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run inference on xView3 model."
    )

    parser.add_argument("--image_folder", help="Path to the xView3 images")
    parser.add_argument("--scene_ids", help="Comma separated list of test scene IDs", default=None)
    
    parser.add_argument("--class_model_path", help="Path to trained model")
    parser.add_argument("--class_channel_indexes", nargs="+", default=[0,1], type=int)

    parser.add_argument("--length_model_path", help='path to lenght model')
    parser.add_argument("--length_channel_indexes", nargs="+", default=[0,1], type=int)
    
    parser.add_argument("--cfar_configs_name", help="Path to cfar configs", default='base_configs.yml')
    parser.add_argument("--cfar_output_root", help="CFAR Output Root", default='predictions/cfar')
    
    parser.add_argument("--output", help="Path in which to output inference CSVs", default='predictions/predictions.csv')
        
    parser.add_argument("--overwrite", help="CFAR Output Root", dest='overwrite', action='store_true')
    parser.add_argument("--save_cfar", help="Save Intermediate CFAR", dest='save_cfar', action='store_true')
    parser.add_argument("--invert_scene_list", help="invert_scene list", dest='invert_scene_list', action='store_true', default=False)
    parser.add_argument("--shuffle_scene_list", help="invert_scene list", dest='shuffle_scene_list', action='store_true', default=False)
    args = parser.parse_args()
    print(args)
#     while True:
    main(args)
#         threading.sleep(5000)

