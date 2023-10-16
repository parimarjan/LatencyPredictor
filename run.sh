
#CONFIG_DIR=$1
#LR=$2
#DECAY=$3
#CONFIG=configs/evals/single/config_noimdb_mixed.yaml
#CONFIG=configs/evals/single/config_noimdb_debug.yaml
CONFIG=configs/evals/all/config_noimdb.yaml
bash run_latency.sh $CONFIG 0.00001 0.1 1 300
bash run_latency.sh $CONFIG 0.00001 0.1 2 300
bash run_latency.sh $CONFIG 0.00001 0.1 4 300
bash run_latency.sh $CONFIG 0.00001 0.1 8 300
bash run_latency.sh $CONFIG 0.00001 0.1 16 300

#bash run_latency.sh configs/evals/single/config_noimdb.yaml 0.00001 0.1 4 300
#bash run_latency.sh configs/evals/single/config_noimdb.yaml 0.0001 0.0 1 300
#bash run_latency.sh configs/evals/single/config_noimdb.yaml 0.0001 0.1 1 300
#bash run_latency.sh configs/evals/single/config_noimdb.yaml 0.00001 0.0 1 300
#bash run_latency.sh configs/evals/single/config_noimdb.yaml 0.0001 0.0 2 300
#bash run_latency.sh configs/evals/single/config_noimdb.yaml 0.0001 0.0 3 300
#bash run_latency.sh configs/evals/single/config_noimdb.yaml 0.0001 0.0 4 300

#bash run_latency.sh configs/evals/single/config_noimdb.yaml 0.0001 0.1 2 300
#bash run_latency.sh configs/evals/single/config_noimdb.yaml 0.00001 0.0 2 300

#bash run_latency.sh configs/evals/configs/evals/single/config_noimdb.yaml 0.0001 0.0 2 300
#bash run_latency.sh configs/evals/configs/evals/single/config_noimdb.yaml 0.0001 0.1 2 300
#bash run_latency.sh configs/evals/configs/evals/single/config_noimdb.yaml 0.00001 0.0 2 300

#bash run_latency.sh configs/evals/config_nostack.yaml 0.00005 0.0 1
#bash run_latency.sh configs/evals/config_nostack.yaml 0.00005 0.0 2
#bash run_latency.sh configs/evals/config_nostack.yaml 0.00005 0.0 3
#bash run_latency.sh configs/evals/config_nostack.yaml 0.00005 0.0 4

