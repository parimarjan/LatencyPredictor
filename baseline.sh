#python3 main.py --config configs/baselines/config_dbms_stack.yaml \
  #--alg dbms --wandb_tags baseline-final --test_size 0.0

python3 main.py --config configs/baselines/config_dbms_stack.yaml \
  --alg dbms-all --wandb_tags baseline-final --test_size 0.0

#python3 main.py --config configs/baselines/config_dbms_imdb.yaml \
  #--alg dbms --wandb_tags baseline-final-imdb --test_size 0.0
