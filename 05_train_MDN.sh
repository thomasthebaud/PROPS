K=4
dataset_name="Capspeech_1000"
train_fraction=1
echo "using $K components" 


srun -p gpu --gpus 1 python bin/train.py \
    --epochs 10 \
    --batch-size 256 \
    --hidden-dims '1024,2048,1024' \
    --num-components $K \
    --device cuda \
    --output exp/MDN_models/${K}_components_${dataset_name}_p=${train_fraction}.pt \
    --dataset-name "$dataset_name" \
    --train-fraction "$train_fraction" \
    --profile-csv-path data/${dataset_name}/profile_prompts.csv \
    --sbert-root exp/SBERT_embs/${dataset_name} \
    --gmm-init-path "exp/GMMs/${K}_components_precomputed" \
    --entropy-weight 0 \
    --mean-norm-weight 0 \
    --variance-down-weight 0 \
    --pi-regularisation-weight 0 \
    --pi-profile-ce-weight 1.0 \
    --lr 0.0001 \
    --log-interval 1 \
    --model-name "ComposedGMM_MDN"



#,GigaSpeech_train=0

