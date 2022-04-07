

CFAROUTPUT=predictions/cfar_ds.2
LENGTHMODEL=models/length_CNN_logloss_fulldata.h5
MODELPATH=models/VGG-FalsePositives-3.h5
OUTPUTPATH=predictions/validation-VGG_predictions_ds2_4.csv
CONFIGS_FILENAME=base_configs_downsample.2.yml
python predict_image.py --image_folder "$INPUT_IMAGE_FOLDER_HERE" --class_model_path "$MODELPATH" --class_channel_indexes 0 1 --length_model_path "$LENGTHMODEL" --length_channel_indexes 0 1 3 --save_cfar --cfar_output_root "$CFAROUTPUT" --output "$OUTPUTPATH" --cfar_configs_name="$CONFIGS_FILENAME"


