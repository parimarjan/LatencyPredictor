CONFIG_DIR=$1
LR=$2
DECAY=$3
INSTANCES=$4

ARCH=gcn

python3 main.py --config ${CONFIG_DIR} --num_instances $INSTANCES --wandb_tags final-min-01 \
  --seed 1 --eval_epoch 5 --num_epochs 100 --arch $ARCH --lr $LR --weight_decay $DECAY &
python3 main.py --config ${CONFIG_DIR} --num_instances $INSTANCES --wandb_tags final-min-01 \
  --seed 2 --eval_epoch 5 --num_epochs 100 --arch $ARCH --lr $LR --weight_decay $DECAY &
python3 main.py --config ${CONFIG_DIR} --num_instances $INSTANCES --wandb_tags final-min-01 \
  --seed 3 --eval_epoch 5 --num_epochs 100 --arch $ARCH --lr $LR --weight_decay $DECAY &
python3 main.py --config ${CONFIG_DIR} --num_instances $INSTANCES --wandb_tags final-min-01 \
  --seed 4 --eval_epoch 5 --num_epochs 100 --arch $ARCH --lr $LR --weight_decay $DECAY &
python3 main.py --config ${CONFIG_DIR} --num_instances $INSTANCES --wandb_tags final-min-01 \
  --seed 5 --eval_epoch 5 --num_epochs 100 --arch $ARCH --lr $LR --weight_decay $DECAY &

ARCH2=factorized
python3 main.py --config ${CONFIG_DIR} --num_instances $INSTANCES --wandb_tags final-min-01 \
  --seed 1 --eval_epoch 5 --num_epochs 100 --arch $ARCH2 --lr $LR --weight_decay $DECAY &
python3 main.py --config ${CONFIG_DIR} --num_instances $INSTANCES --wandb_tags final-min-01 \
  --seed 2 --eval_epoch 5 --num_epochs 100 --arch $ARCH2 --lr $LR --weight_decay $DECAY &
python3 main.py --config ${CONFIG_DIR} --num_instances $INSTANCES --wandb_tags final-min-01 \
  --seed 3 --eval_epoch 5 --num_epochs 100 --arch $ARCH2 --lr $LR --weight_decay $DECAY &
python3 main.py --config ${CONFIG_DIR} --num_instances $INSTANCES --wandb_tags final-min-01 \
  --seed 4 --eval_epoch 5 --num_epochs 100 --arch $ARCH2 --lr $LR --weight_decay $DECAY &
python3 main.py --config ${CONFIG_DIR} --num_instances $INSTANCES --wandb_tags final-min-01 \
  --seed 5 --eval_epoch 5 --num_epochs 100 --arch $ARCH2 --lr $LR --weight_decay $DECAY

#python3 main.py --config ${CONFIG_DIR} --num_instances 2 \
  #--seed 1 --eval_epoch 50 --num_epochs 100 &
#python3 main.py --config ${CONFIG_DIR} --num_instances 2 \
  #--seed 2 --eval_epoch 50 --num_epochs 100 &
#python3 main.py --config ${CONFIG_DIR} --num_instances 2 \
  #--seed 3 --eval_epoch 50 --num_epochs 100 &

#python3 main.py --config ${CONFIG_DIR} --num_instances 3 \
  #--seed 1 --eval_epoch 50 --num_epochs 100 &
#python3 main.py --config ${CONFIG_DIR} --num_instances 3 \
  #--seed 2 --eval_epoch 50 --num_epochs 100 &
#python3 main.py --config ${CONFIG_DIR} --num_instances 3 \
  #--seed 3 --eval_epoch 50 --num_epochs 100 &

#python3 main.py --config ${CONFIG_DIR} --num_instances 4 \
  #--seed 1 --eval_epoch 50 --num_epochs 100 &
#python3 main.py --config ${CONFIG_DIR} --num_instances 4 \
  #--seed 2 --eval_epoch 50 --num_epochs 100 &
#python3 main.py --config ${CONFIG_DIR} --num_instances 4 \
  #--seed 3 --eval_epoch 50 --num_epochs 100

sleep 1000
