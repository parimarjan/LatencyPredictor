CONFIG_DIR=$1
LR=$2
DECAY=$3
INSTANCES=$4
EPOCHS=$5
EXTRA=$6
YLOG=$7

TAG=single-testq-final6-fixed_instances-min-01

ARCH2=factorized

python3 main.py --config ${CONFIG_DIR} --num_instances $INSTANCES --wandb_tags $TAG \
  --seed 1 --eval_epoch 1000 --num_epochs $EPOCHS --arch $ARCH2 --lr $LR \
  --weight_decay $DECAY --latent_inference 0 --extra_training $EXTRA --log_transform_y $YLOG &
python3 main.py --config ${CONFIG_DIR} --num_instances $INSTANCES --wandb_tags $TAG \
  --seed 2 --eval_epoch 1000 --num_epochs $EPOCHS --arch $ARCH2 --lr $LR \
  --latent_inference 0 --weight_decay $DECAY --extra_training $EXTRA --log_transform_y $YLOG &
python3 main.py --config ${CONFIG_DIR} --num_instances $INSTANCES --wandb_tags $TAG \
  --seed 3 --eval_epoch 1000 --num_epochs $EPOCHS --arch $ARCH2 --lr $LR \
  --latent_inference 0 --weight_decay $DECAY --extra_training $EXTRA \
  --log_transform_y $YLOG &
python3 main.py --config ${CONFIG_DIR} --num_instances $INSTANCES --wandb_tags $TAG \
  --seed 4 --eval_epoch 1000 --num_epochs $EPOCHS --arch $ARCH2 --lr $LR \
  --latent_inference 0 --weight_decay $DECAY --extra_training $EXTRA \
  --log_transform_y $YLOG
