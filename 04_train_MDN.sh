K=128
CV_p=0.05
GS_p=0
echo "using $K components" 


srun -p gpu --gpus 1  python bin/train.py \
    --epochs 100 \
    --batch-size 256 \
    --hidden-dims '1024,2048,2048,1024' \
    --num-components $K \
    --device cuda \
    --output exp/MDN_models/${K}_components_CV=${CV_p}_GS=${GS_p}.pt \
    --train-csv-paths "CommonVoice_train" \
    --train-dataset-fractions CommonVoice_train=${CV_p} \
    --dev-csv-paths "CommonVoice_dev" \
    --profile-csv-path data/profile_prompts.csv \
    --entropy-weight 1 \
    --mean-norm-weight 0 \
    --variance-down-weight 0 \
    --pi-regularisation-weight 0.1 \
    --lr 0.0005 \
    --log-interval 1 \
    --model-name "GaussianMDN"



#,GigaSpeech_train=0

