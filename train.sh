python train.py \
    --cuda \
    -d voc \
    -root /home/yifux/ \
    --batch_size 16 \
    --img_size 512 \
    --backbone r50 \
    --max_epoch 150 \
    --lr_drop 100 \
    --eval_epoch 2 \
    --aux_loss \
    --use_nms \
    --no_warmup \
    # --start_epoch 0 \
    # --resume weights/voc/xxx.pth \
    # --mlp_dim 2048 \
    # --hidden_dim 256 \
    # --batch_first
