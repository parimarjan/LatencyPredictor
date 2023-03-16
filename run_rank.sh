#EMBS=(1 2 4 8 16 32 64 128)
EMBS=(128)
REPS=(1)

for ri in "${REPS[@]}"
  do
  for es in "${EMBS[@]}"
    do
      CMD="python3 main.py --config configs/rank_configs/config.yaml \
      -es $es --num_epochs 200
      "
      echo $CMD
      eval $CMD
  done
done
