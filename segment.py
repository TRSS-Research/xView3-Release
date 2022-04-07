import numpy as np
import pandas as pd
import multiprocessing as mp
from src.object_detection import utils
import datetime as dt
from src.object_detection.object_detection_cfar import process_scene_cfar_handler, cfar_inference
from src.object_detection.subset_export_images import process_scene_true_positives
from typing import Union, Callable
from functools import partial
import fire
import os
today = str(dt.date.today())


def process_scene_list(scene_list, process_scene: Callable, name=today,
                       data_folder_name='training', overwrite: bool=False):
    """
        run algorithm on each scene image
    :param scene_list:  list of scene hashes
    :param name: identifying string for output folder
    :param configs_name: filename for configs yaml
    :param training_or_validation: either training or validation, for identifying folders
    :param process_scene: algorithm function to segement  the image
    :param overwrite: overwrite previous runs!
    :return:
    """
    logger = utils.get_logger(process="ImageSegmentation", logger_name=f'process_scene_list: {name}')
    n_scenes = len(scene_list)
    for i, scene_id in enumerate(scene_list):
        if os.path.exists(f'output/{name}/{data_folder_name}/{scene_id}/X_0.npy') and not overwrite:
            continue
        logger.info(f'Beginning scene #{i+1}/{n_scenes}: {scene_id}')
        try:
            process_scene(scene_id=scene_id)
        except utils.AWSCorruptedError:
            logger.warning(f'Failed with {scene_id}, Corrupted file in AWS')


def cfar_for_prediction(scene_list, data_root, cfar_configs_name, cfar_output_root, overwrite: bool=False, processes=1):
    if processes == 1:
        for scene_id in scene_list:
            cfar_worker((scene_id, data_root, cfar_configs_name, cfar_output_root, overwrite))
    else:
        work = [(scene_id, data_root, cfar_configs_name, cfar_output_root, overwrite) for scene_id in scene_list]
        with mp.Pool(processes) as pool:
            pool.map(cfar_worker, work)


def cfar_worker(args):
    """
        multiprocessing worker function for CFAR processing
    :param args: (scene_id, data_root, cfar_configs_name, cfar_output_root, overwrite)
    :return:
    """
    scene_id, data_root, cfar_configs_name, cfar_output_root, overwrite = args
    cfar_output_folder = os.path.join(cfar_output_root, scene_id)
    centroids_filepath = os.path.join(cfar_output_folder, 'centroids.npy')
    X_filepath = os.path.join(cfar_output_folder, 'X.npy')
    length_filepath = os.path.join(cfar_output_folder, 'length.npy')
    if not all((os.path.exists(centroids_filepath), os.path.exists(X_filepath), os.path.exists(length_filepath))) \
            or overwrite:
        print(f'Working on {scene_id}')
        try:
            X, centroids, length = cfar_inference(
                data_folder=os.path.join(data_root, scene_id), configs_name=cfar_configs_name)
        except utils.AWSCorruptedError:
            print(f'Failed with {scene_id}, Corrupted file in AWS')
        else:
            utils.save_npy(centroids_filepath, centroids)
            utils.save_npy(X_filepath, X)
            utils.save_npy(length_filepath, length)


def process_from_source(from_s3: bool, data_folder_name: str, name: str, process_scene: Callable,
                        limit=(0, 20), random_scenes=False, overwrite=False):
    if isinstance(limit, int):
        limit = (0, limit)
    if from_s3:

        scene_list = utils.get_scene_list_s3(data_folder_name, limit=limit, random=random_scenes)
    else:
        scene_list = utils.get_scene_list_folder(f'data/{data_folder_name}', limit=limit, random=random_scenes)
    process_scene_list(scene_list, process_scene=process_scene, data_folder_name=data_folder_name, name=name,
                       overwrite=overwrite)


def process_scene_cli(name: str, data_folder_name: str = 'training' , cfar: bool = True, from_s3: bool = True,
                      limit: Union[tuple, int] = 20, config_name: str = 'base_configs.yml', upload: bool=False,
                      random_scenes: bool=False, overwrite: bool=False):
    """
        
    :param name: Name of job. Will be used for saving the files
    :param data_folder_name: If true, run on 'training' data
    :param cfar:  if true, run CFAR algorithm, else run segmentation process using true-positives
    :param from_s3: if true, gather data from s3
    :param limit: number of scenes to process. If int, the upper number, if tuple, the lower and upper index
    :param config_name: config file options for algorithm
    :param upload: if true, load to s3
    :param random_scenes: if true, randomize scenes returned
    :return:
    """
    name = str(name)
    if cfar:
        name += '_cfar'
        process_scene = partial(process_scene_cfar_handler, jobname=name, folder_name=data_folder_name,
                                s3=from_s3, configs_name=config_name)
    else:
        name += '_segment'
        process_scene = partial(process_scene_true_positives, get_negative_examples=True, plot=False,
                                jobname=name, configs_name=config_name, data_folder_name=data_folder_name)


    process_from_source(from_s3=from_s3,
                        process_scene=process_scene,
                        data_folder_name=data_folder_name,
                        name=name,
                        limit=limit,
                        random_scenes=random_scenes,
                        overwrite=overwrite,
                        )

    if upload:
        utils.load_to_s3(name=name, data_folder_name=data_folder_name)



def cfar_for_prediction_cli(image_folder: str, configs_name: str, output_root: str, overwrite:bool = False,
                            processes: int=1):
    from predict_image import get_scenelist
    scene_list = get_scenelist(image_folder)
    cfar_for_prediction(scene_list=scene_list, data_root=image_folder, cfar_configs_name=configs_name,
                           cfar_output_root=output_root, overwrite=overwrite, processes=processes)


if __name__ == '__main__':
    fire.Fire({
        'process_scene': process_scene_cli,
        'to_s3': utils.load_to_s3,
        'cfar_for_prediction_mp': cfar_for_prediction_cli
    })