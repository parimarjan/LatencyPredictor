CONFIG_DIR=$1
python3 main.py --config ${CONFIG_DIR} --num_instances 1 \
  --seed 1 --eval_epoch 50 --num_epochs 200 &
python3 main.py --config ${CONFIG_DIR} --num_instances 1 \
  --seed 2 --eval_epoch 50 --num_epochs 200 &
python3 main.py --config ${CONFIG_DIR} --num_instances 1 \
  --seed 3 --eval_epoch 50 --num_epochs 200 &

python3 main.py --config ${CONFIG_DIR} --num_instances 2 \
  --seed 1 --eval_epoch 50 --num_epochs 200 &
python3 main.py --config ${CONFIG_DIR} --num_instances 2 \
  --seed 2 --eval_epoch 50 --num_epochs 200 &
python3 main.py --config ${CONFIG_DIR} --num_instances 2 \
  --seed 3 --eval_epoch 50 --num_epochs 200 &

python3 main.py --config ${CONFIG_DIR} --num_instances 3 \
  --seed 1 --eval_epoch 50 --num_epochs 200 &
python3 main.py --config ${CONFIG_DIR} --num_instances 3 \
  --seed 2 --eval_epoch 50 --num_epochs 200 &
python3 main.py --config ${CONFIG_DIR} --num_instances 3 \
  --seed 3 --eval_epoch 50 --num_epochs 200 &

python3 main.py --config ${CONFIG_DIR} --num_instances 4 \
  --seed 1 --eval_epoch 50 --num_epochs 200 &
python3 main.py --config ${CONFIG_DIR} --num_instances 4 \
  --seed 2 --eval_epoch 50 --num_epochs 200 &
python3 main.py --config ${CONFIG_DIR} --num_instances 4 \
  --seed 3 --eval_epoch 50 --num_epochs 200
sleep 1000
