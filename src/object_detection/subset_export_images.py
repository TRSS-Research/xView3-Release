import numpy as np
import os
import pandas as pd
from src  import static
from src.object_detection import utils
import numba
root = os.getcwd()
negative_example_modifier = 5


# shitf_x / y as a function of output image size
def process_scene_true_positives(scene_id: str, jobname: str, configs_name='base_configs.yml',
                                 get_negative_examples=True, plot=False, data_folder_name='training'
                                 ) -> (static.MetadataDataFrame, static.TrainData, static.TestData):
    """
        export a stack of images centered on the centroids provided in the metadata.
        optionally augment the data to move around the center pixel and get different 'windows' onto the object.
        Optinally add negative examples of 'open ocean'
    :param scene_id:
    :param jobname: name of this processing job
    :param configs_name: configs filename
    :param get_negative_examples: True to get negative examples too
    :param plot: output images
    :param data_folder_name: input folder above 'sceneid'
    :return:
    """

    configs = utils.load_configs(os.path.join(root, 'configs', 'segmentation', configs_name))
    assert configs['shifts']['x'] < .5
    assert configs['shifts']['y'] < .5

    shift_x = int(configs['shifts']['x'] * configs['windows']['window_x'])
    shift_y = int(configs['shifts']['y'] * configs['windows']['window_y'])

    logger = utils.get_logger(process="ImageSegmentation", logger_name='process_scene_true_positives-' + scene_id)
    logger.info(f'Begin scene {scene_id} with config {configs_name}')

    paths_dict = utils.get_paths(root, training_or_validation=data_folder_name, scene_id=scene_id)
    logger.info('Loading All Image Data')
    named_data = utils.load_image_data_from_paths_dict(paths_dict, configs['resampling'])
    logger.info('Completed Loading Image Data')

    df = utils.load_subset_data(filepath=paths_dict[data_folder_name], scene_id=scene_id)
    df = df.loc[pd.notnull(df['is_vessel']), :] # only want high quality segments, discard other data
    df['label'] = df.apply(utils.get_label_apply, axis=1)

    logger.info('Augmenting Data')
    label_row_cols = df[['detect_scene_row', 'detect_scene_column']].values
    label_row_cols_augment = augment_label_rows(label_row_cols, configs['n_shifts'], shift_x=shift_x, shift_y=shift_y)

    logger.info('Calculate Valid Masks')
    valid_water_mask = utils.get_valid_water_mask(named_data['mask'], named_data['VV'])

    # Negative Examples
    n_negative_examples = 0
    if get_negative_examples:
        n_negative_examples = configs['n_negative_examples']
        logger.info('Seeding Negative Examples')
        negative_examples = get_negatives(true_row_cols=label_row_cols, valid_water_mask=valid_water_mask,
                                          n=n_negative_examples, **configs['windows'])
        label_row_cols_augment = np.vstack((label_row_cols_augment, negative_examples))

    logger.info(f'Collecting Centroids')
    X = utils.subset_image_arrays(centroids=label_row_cols_augment, named_data_dict=named_data, **configs['windows'])

    df_aug = pd.concat([df] * configs['n_shifts'], ignore_index=True)
    print('AugShape', df_aug.shape)
    df_aug.loc[:, 'augment-detect_scene_row'] = label_row_cols_augment[: -n_negative_examples, 0]
    df_aug.loc[:, 'augment-detect_scene_column'] = label_row_cols_augment[: -n_negative_examples, 1]
    df_aug.loc[:, 'negative_example'] = False
    if get_negative_examples:
        data_negative = [[None]*len(df.columns) for _ in range(n_negative_examples)]

        df_neg = pd.DataFrame(data_negative, columns=df.columns)
        df_neg.loc[:, 'is_vessel'] = False
        df_neg.loc[:, 'is_fishing'] = False
        df_neg.loc[:, 'detect_scene_row'] = negative_examples[:, 0]
        df_neg.loc[:, 'detect_scene_column'] = negative_examples[:, 1]
        df_neg.loc[:, 'augment-detect_scene_row'] = df_neg.loc[:, 'detect_scene_row']
        df_neg.loc[:, 'augment-detect_scene_column'] = df_neg.loc[:, 'detect_scene_column']
        df_neg.loc[:, 'negative_example'] = True
        df_neg.loc[:, 'label'] = 'NonObject'
        df_aug = df_aug.append(df_neg,ignore_index=True)
    df_aug['augmentation-index'] = np.arange(df_aug.shape[0])
    y = df_aug['label'].astype(static.cat_type).cat.codes.values

    if configs.get('save_intermediate', False):
        logger.info(f'Saving Files')
        data_output_folder = os.path.join(root, 'output', jobname, data_folder_name)
        utils.save_intermediate_separate(data_output_folder, scene_id, df_aug, X, y)

    if plot:
        image_output_folder = os.path.join(root, 'output', data_folder_name, scene_id, 'augment-images')
        os.makedirs(image_output_folder, exist_ok=True)

        logger.info(f"Plotting Images to {image_output_folder}")

        choices = np.random.choice(np.arange(label_row_cols_augment.shape[0]), replace=False, size=10)
        utils.plot_images(row_cols=label_row_cols_augment[choices, :],
                          named_data_dict=named_data, centroid_labels=y[choices],
                          alt_centers=np.tile(label_row_cols_augment.T, reps=configs['n_shifts']).T[choices, :],
                          root_out=image_output_folder)
    return df_aug, (X, y)


def augment_label_rows(centroids: static.Centroids, n: int, shift_x: int, shift_y: int):
    """
        duplicate number of examples by shifting the window around the pixel
    :param centroids: list/array of centroids
    :param n: number of augmentations per true label
    :param shift_x: maximum x shift pixel count
    :param shift_y: maximum y shift pixel count
    :return:
    """
    groups = []
    for i in range(n):
        x_shifts = np.random.randint(low=-shift_x, high=shift_x, size=centroids.shape[0])
        y_shifts = np.random.randint(low=-shift_y, high=shift_y, size=centroids.shape[0])
        shifts = np.stack((y_shifts, x_shifts)).T
        groups.append(centroids + shifts)
    group = np.vstack(groups)
    return group


@numba.jit(forceobj=True)
def get_negatives(true_row_cols: np.ndarray, valid_water_mask: np.ndarray,
                  n:int, window_x:int, window_y:int):
    negatives_mask = np.ones(shape=valid_water_mask.shape, dtype=np.bool)
    for i, j in true_row_cols:
        negatives_mask[i - window_y:i + window_y, j - window_x:j + window_x] = 0
    open_ocean_mask = np.logical_and(negatives_mask, valid_water_mask)
    potentials = np.random.randint(valid_water_mask.shape, size=(n * 50, 2))
    chosen_ix = np.where(open_ocean_mask[potentials[:, 0], potentials[:, 1]])[0][:n]

    return potentials[chosen_ix]


if __name__ == '__main__':
    process_scene_true_positives('00a035722196ee86t',plot=True)