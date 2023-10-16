CONFIG_DIR=$1
ARCH=$2
TAG=$3

python3 main.py --wandb_tags $TAG --config $CONFIG_DIR --arch $2 \
  --eval_epoch 100 --num_epochs 50 --lr 0.0001
#python3 main.py --wandb_tags $TAG --config $CONFIG_DIR --arch $2 \
  #--eval_epoch 100 --num_epochs 50 --lr 0.0001

#python3 main.py --wandb_tags query_split2 --config $CONFIG_DIR \
  #--eval_epoch 100 --num_epochs 50 --lr 0.0001 &
#python3 main.py --wandb_tags query_split2 --config $CONFIG_DIR --arch gcn \
  #--eval_epoch 100 --num_epochs 50 --lr 0.0001 &
#python3 main.py --wandb_tags query_split2 --config $CONFIG_DIR \
  #--eval_epoch 100 --num_epochs 50 --lr 0.0001


#sleep 100
