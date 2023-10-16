

python3 train_linux.py --config configs/pretrain/config_linux.yaml \
 --lr 0.0001 --num_epochs 5000 --eval_epoch 5 &

python3 train_linux.py --config configs/pretrain/config_linux.yaml \
  --lr 0.001 --num_epochs 5000 --eval_epoch 5 &

python3 train_linux.py --config configs/pretrain/config_linux.yaml \
  --lr 0.00005 --num_epochs 5000 --eval_epoch 5
