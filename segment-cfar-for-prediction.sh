# edit the variables here!
imagefolder=$1
configsname=$2
outputroot=$3

python segment.py cfar_for_prediction_mp --image_folder "$imagefolder" --output_root "$outputroot" --configs_name="$configsname" --processes=10