#!/usr/bin/env zsh
 

echo "*********************************************"
echo "loading done" 
echo "*********************************************"

target_dir='../Images_Data/CV_10_Folding/Fold_1/temp_train_lm/'
input_dir='../Images_Data/CV_10_Folding/Fold_1/temp_train_png/'
checkpoint='../Images_Data/CV_10_Folding/Fold_1/Models/Models_flat/Models_lm_5/'
output_dir='../Images_Data/CV_10_Folding/Fold_1/Models/Models_flat/Models_lm_5/'
landmarks='2,12,31,41,41'
python3 Pr_LandMarkDetection_FlatArc+HeatMap.py --mode 'train'   --input_dir   $input_dir     --target_dir  $target_dir    --checkpoint  $checkpoint     --output_dir  $output_dir     --landmarks  $landmarks 

target_dir='../Images_Data/CV_10_Folding/Fold_1/temp_test_lm/'
input_dir='../Images_Data/CV_10_Folding/Fold_1/temp_test_png/'
checkpoint='../Images_Data/CV_10_Folding/Fold_1/Models/Models_flat/Models_lm_5/'
output_dir='../Images_Data/CV_10_Folding/Fold_1/Results/Models_flat/Models_lm_5/'
landmarks='2,12,31,41,41'
python3 Pr_LandMarkDetection_FlatArc+HeatMap.py --mode 'test'   --input_dir   $input_dir     --target_dir  $target_dir    --checkpoint  $checkpoint     --output_dir  $output_dir     --landmarks  $landmarks 


echo "*********************************************"
echo "python job done" 
### rm -rf ../Images_Data/CV_10_Folding/Fold_1/Models_flat/Models_lm_4/
echo "model removed" 
echo "*********************************************"

