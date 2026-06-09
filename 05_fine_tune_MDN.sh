K=4
CV_p=1
GS_p=0
echo "using $K components" 


srun -p gpu --gpus 1 python bin/train.py \
    --epochs 100 \
    --batch-size 256 \
    --hidden-dims '1024,2048,2048,1024' \
    --num-components $K \
    --device cuda \
    --output exp/MDN_models/${K}_components_CV=${CV_p}.pt \
    --train-csv-paths "CommonVoice_train" \
    --train-dataset-fractions CommonVoice_train=${CV_p} \
    --dev-csv-paths "CommonVoice_dev" \
    --profile-csv-path data/profile_prompts.csv \
    --entropy-weight 0 \
    --mean-norm-weight 0 \
    --variance-down-weight 0 \
    --pi-regularisation-weight 0 \
    --lr 0.0001 \
    --log-interval 1 \
    --model-name "ComposedGMM_MDN"



#,GigaSpeech_train=0

