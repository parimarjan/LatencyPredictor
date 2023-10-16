
#bash run_query_split.sh configs/query_splits/stack/config_trans.yaml factorized
#bash run_query_split.sh configs/query_splits/stack/config_hist.yaml factorized

#bash run_query_split.sh configs/query_splits/stack/config_trans.yaml factorized $TAG

#bash run_query_split.sh configs/query_splits/stack/config_hist.yaml factorized $TAG
#bash run_query_split.sh configs/query_splits/stack/config_sys.yaml factorized $TAG

TAG=stack_query_split4
bash run_query_split.sh configs/query_splits/stack/config_hist.yaml gcn $TAG
#bash run_query_split.sh configs/query_splits/stack/config_sys.yaml factorized $TAG
#bash run_query_split.sh configs/query_splits/stack/config_hist.yaml factorized $TAG
#bash run_query_split.sh configs/query_splits/stack/config_trans.yaml factorized $TAG



#bash run_query_split.sh configs/query_splits/imdb/config_trans.yaml dbms query_split_imdb
