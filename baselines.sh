python3 main.py --config configs/hyp_pretrain/config_all.yaml --sys_net_arch mlp
python3 main.py --config configs/hyp_pretrain/config_all.yaml \
	--sys_net_arch avg --factorized_net_arch mlp
