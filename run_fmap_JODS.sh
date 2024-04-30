#!/bin/bash
# preprocess 

dino_save_path="data-tss/TSS_CVPR2016/JODS_feat_token_11_NOMASK"
sd_save_path="data-tss/TSS_CVPR2016/JODS_sd"
fmap_save_path="data-tss/TSS_CVPR2016/JODS_fmap"
fmap_save_path_2="data-tss/TSS_CVPR2016/JODS_fmap_2"


python preprocess.py --dataset_name JODS \
    --root_dir_path data-tss/.s/TSS_CVPR2016/JODS \
    --facet token \
    --layer 11 \
    --save_path data-tss/TSS_CVPR2016/JODS_feat_token_11_NOMASK \
    --root_path data-tss/.s/TSS_CVPR2016/ 

echo "Done extracting DINO features"

python tss_sd.py --dataset JODS\
                --root_dir_path data-tss/.s/TSS_CVPR2016/JODS \
                --save_path data-tss/TSS_CVPR2016/JODS_sd 

echo "Done extracting SD features"

# calculate fmap 
device_ids=(0) # maybe you can use more devices
cases=($(seq 0 80))

for case in "${cases[@]}"; do
    for device_id in "${device_ids[@]}"; do

        new_device_id=$((device_id + case * 1))

        save_dir="${new_device_id}_tss"

        root=$sd_save_path
        root_cons=$dino_save_path

        python train.py \
                --device_id $device_id \
                --save_dir $fmap_save_path/$save_dir \
                --root $root \
                --case $case \
                --TSS True \
                --root_cons $root_cons &

            # Optionally, you can add more arguments as needed
        

        # Wait for all jobs for the current category and case to finish before moving to the next case
        
    done

    wait
done

echo "Done calculating FMAP"

# run evaluation

python eval_tss.py  \
    --SUBSET JODS \
    --fmap_path data-tss/TSS_CVPR2016/JODS_fmap/ \
    --DINO_path data-tss/TSS_CVPR2016/JODS_feat_token_11_NOMASK \
    --SD_path data-tss/TSS_CVPR2016/JODS_sd \
    --FMAP True \
    --SD False 

echo "Done evaluating FMAP"

# calculate fmap 
device_ids=(0) # maybe you can use more devices
cases=($(seq 0 80))

for case in "${cases[@]}"; do
    for device_id in "${device_ids[@]}"; do

        new_device_id=$((device_id + case * 1))

        save_dir="${new_device_id}_tss"

        root_cons=$sd_save_path
        root=$dino_save_path

        python train.py \
                --device_id $device_id \
                --save_dir $fmap_save_path_2/$save_dir \
                --root $root \
                --case $case \
                --TSS True \
                --root_cons $root_cons &

            # Optionally, you can add more arguments as needed
        

        # Wait for all jobs for the current category and case to finish before moving to the next case
        
    done

    wait
done

echo "Done calculating FMAP"

# run evaluation

python eval_tss.py  \
    --SUBSET JODS \
    --fmap_path data-tss/TSS_CVPR2016/JODS_fmap_2/ \
    --DINO_path data-tss/TSS_CVPR2016/JODS_feat_token_11_NOMASK \
    --SD_path data-tss/TSS_CVPR2016/JODS_sd \
    --FMAP True \
    --SD True 