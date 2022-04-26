
# 7th Place Solution for xView3 Challenge (2021)

This repo contains the code developed by the Thomson Reuters Special Services team for the xView3 challenge, 
predicting Illegal, Unreported and Unregulated fishing vessels. It placed 28th in leaderboard performance, but among
verified submissions (i.e. those that met the compute restrictions of the project), it was 7th. 
The performance on the public holdout set was 0.42, and was very similar on the private holdout set. 

Public-Holdout Performance     | Overall Score | Object Detection F1       | Close-to-Shore Detection F1 | Vessel Classification F1      | Fishing Vessel Classification F1      | Length Estimation
------|-------|--------|--------|-------|-----------|-----------
Score | 0.42108 |	0.61164	|0.15181	|0.91565|	0.75209|	0.62265	

Model prediction was relatively quick, around 10 minutes per prediction on the commodity hardware equivalent employed by the xView team.   

Please see a detailed write up here:

https://github.com/TRSS-Research/xView3-Release/blob/master/Xview3_Release.pdf

## Task

The third installment of the xView series focused on IUU fishing. The project organizers collected, 
cleaned, standardized and labeled over 500 SAR scenes with two polarizations (VV, VH), as well as accompanying 
Wind, Bathymetry and Land / Ice data.

## Approach 


Our team arrived at a three component model solution:  an object detection module, a classification module, and a length estimation module.

This was the primary team members' first foray into both neural models and computer vision, and we were initially 
intimidated by the massive scale of the images, so we decided to split the work into these subunits. 
We thought that neural models would not be able to segment on these gigapixel images without preprocessing, and believed 
(and found) a simpler, multi-class classification would be able to discriminate well against different kinds of objects.

- The object detection piece is an adaptation of the CFAR algorithm, well used in SAR imaging projects (citation), implemented 
in a serial multiresolution pipeline. 
- For classification we leveraged the VGG16 architecture and trained it from scratch on labeled sub-images, centered on the apparent non-object/non-vessel/non-fishing-vessel/fishing-vessel, according to the metadata.
- For length estimation, we employed an elementary 5 layer CNN.


## Algorithms/Models

### Object Detection

Our primary object detection component was a Multi Resolution Constant False Alarm Rate (MR-CFAR) algorithm. 
This adaptation of the traditional CFAR algorithm reduces the search space and false positive rate of the full-resolution 
CFAR algorithm. The MRCFAR algorithm differes from CFAR in its passes over the data at different resolutions, with 
each successive pass only looking at the 'anomalous' data raised by previous pass through, decreasing the area of pixels 
needed to test. This exploits the spatial nature of these vessels, passing the collective signal of the entire vessel to 
the next resolution level. 

#### CFAR

CFAR is well documented [[1](https://en.wikipedia.org/wiki/Constant_false_alarm_rate), [2](https://arxiv.org/ftp/arxiv/papers/1804/1804.04719.pdf), [3](https://ieeexplore.ieee.org/document/1411989)], so I will touch on it only briefly. 

CFAR is an algorithm for object detection in RADAR data, exploiting the spatial nature of the targets to expose objects
in the face of noise and interference. The active sensing of the radar system means to pick up reflections of the source beam, 
which will be high in reflective objects, and low otherwise. The alogrithm essentially compares a given observation 
(a pixel for our purposes), against its neighbor pixels, and if the center pixel is much brighter than the
neighbor pixels, we recognize it as an object. 

One complication, as noted, is the spatial nature of the objects reflecting the signal. In order to compare a pixel against
background, and not the other pixels that represent the same object, we obscure the immediate neighborhood of the pixel under
observation, the 'guard' cells, and focus only on the external 'window' pixels. We compute a mean and standard deviation for
all these window pixels, and compare it to the center pixel under observation with a z-score. If the pixel passes a threshold, 
it is labeled an object.

#### Multi Resolution CFAR

We adapted CFAR to be faster and more accurate, by serially applying the algorithm at different resolutions and with
different guard sizes, window sizes, and z-score thresholds. 

We perform a first pass on a low resolution representation of the data, exclusively looking at the water, due to the land/water mask.
With the objects detected in the low resolution pixels, we review a medium resolution. With the objects detected at the 
medium resolution, we to two final passes at full resolution, with different window and guard parameters, exploring the 
windows directly above / below and directly to the sides, to try and reduce any interference from nearby objects. 

In our implementation, the low resolution was at a 10% sample rate, with a 3 cell window and 1 cell guard ,
a 50% sample rate with a 7 cell window and a 4 cell guard, and a 15 cell window an 7 cell guard for the full resolution. 

#### CFAR Post Processing

Once we have our identified the object pixels, we perform postprocessing to find the centroids, and to deduplicate any 
object centroids identified as being too near to one another (within 5 pixels).

With our pixel centroids, we then pass a stack of images centered on the pixels identified to the secondary object 
classification model. 

### Object Classification

We used a convolutional Neuarl Network with the VGG16 architecture (citation) for classifiying the images. We included 
four classes, Non-Object, Non-Vessel, Non-FishingVessel, and Fishing Vessel. 

### Length Estimation 

We used a convolutional neural network for length estimation. 

## Training Data

The MR-CFAR algorithm does not need to be trained, but we did evaluate appropriate configuration parameters by sampling 
scenes and evaluating the object detection F1 scores. 

We specified custom training and validation batches, due to the substantial data quality discrepancy between the
prescribed training and validation data, but maintained scene-level separation. We used 64 x 64 images, with the two 
polarization channels, and did not rely on the supplemental wind and depth data. We scaled the polaizations to 
between 0 and 1 by adding a the minimum value and dividing by a maximum value. We censored the data
to -50 and 20 decibels, so used these as the bounds.


The Object Classification VGG16 model was trained on in two batches. The first batch consisted of all of the Medium and High 
confidence predictions from the xview3 data providers, as well as 100 explicitly negative images per scene. These negative
images were identified as pixels that were more than 40 pixels from an identified low, medium, or high confidence object. 

The second batch consisted of the first batch, with an addition of examples that were false positives from the MR-CFAR object detection 
piece. This was more representative of what the model would need to predict on in the test environment and so greatly 
improved the whole model performance. 

Each training run took about 12 hours on the P1 Sagemaker hardware, which runs a single V100 gpu, with 64 GB of ram.

The length estimation CNN was trained once, on the primary dataset of Medium and High confidence observations. We used 
mean squared error as the loss function, but futher experimentation suggested mean absolute error performed better.  

## Project Structure

The primary code is in the src/ directory, with the MR-CFAR algorithm and related functions can be found in 
[src/object_detection](src/object_detection).

The Classification and Length Estimation CNN model helper functions and data loaders can be found in 
[src/classifier](src/classifier).

### To Train the classification model:



0. Setup
   1. Download Data 
      - You need to first download and unzip the xView3 data into a local folder, specifying validation and training sub directoryies
        - i.e. data/train, data/validation
        - Note: you may use S3 locations instead of local storage if necessary. The Object Detection will process either, 
            but will store the processed subscenes locally
   2. Setup Environment
       - create a conda environment with python 3.6<3.9
       - install dependencies 
         - `conda install rasterio -c conda-forge`
         - `pip install -r requirements.txt`
1. Prepare subsection data for processing
  0. setup configs for processing for segmentation, specifying resampling approaches, and various CFAR
     specific requirements (eg. window sizes, guard sizes, number of passes). 
     - located in `configs/cfar`, and `configs/segmentation`. review the '`config-default.yaml`' for standard values 
  1. run the segmentation script (`segment.py`), specifying the `nocfar` process, the IO directories, and the config file  
     - ex:  `python segment.py process_scene --overwrite --from_s3 --data_folder_name=training --limit=2 --name=demo --nocfar`
  2. run the CFAR object detection script   (`segment.py`), specifying the `cfar` process, the IO Directories and the config file name
     - ex:  `python segment.py process_scene --overwrite --from_s3 --data_folder_name=training --limit=2 --name=demo --cfar`
2. Train the scene classification model
   1. Run the training script `train.py` with specified input directory, epochs, batchsize, output folder. At first,
      specify the directory to correspond with the output from the segmentation script (i.e. `segment.py` with `--nocfar`. 
      This comprises the cleaned training data, with the negative examples (non-object scenes) being intentional open-ocean data
      - ex: `python train.py --directory output/demo_segment --epochs=2` 
   2. Run the training script `train.py` with specified input directory, epochs, batchsize, output folder. This second time, 
      use the same model before (using the `--prebuilt_model_path` option), and point at the output from the CFAR model.
      This data comprises only the positive results from the CFAR segmentaiton process, with the true labels, so misses some of the 
      true positives, but also captures some false positives (that have corrected labels), and will represent the data 
      the model will be predicting on in a much more representative light.
      - ex `python train.py --directory cfar-output/demo_cfar --epochs=2 --prebuilt_model_path=output/models/CFARVGG16/SAR-VGG16.model`
3. Train the length-estimation classification model
    1. Run the training script `train-length.py` with specified input directory, epochs, batchsize, output folder. 
       As this does not depend on the variation from CFAR, we can rely on just the original segmented data. 
       - ex `python train-length.py --directory output/demo_segment --epochs=50`

### To run the trained model ala docker:

- Build:
    - cd into (this) xview3-submission directory
    - `docker build -t xview3 .`
- once built
    - \<absolute_path_to_data> = (eg: C:\Users\username\code\xview3-submission\data\validation)
    - \<image_folder> = data/validation
    - `docker run -v <absolute_path_to_data>:/app/<image_folder> xview3v2 <image_folder> 8204efcfe9f09f94v,844545c005776fb1v predictions_docker.csv`
