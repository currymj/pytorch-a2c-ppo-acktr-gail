python main.py --env-name "Walker2d-v2" --algo wpo --lr 3e-4 --num-steps 1000 --ppo-epoch 20 --num-processes 1 --num-env-steps 300000 --log-interval 1  --log-dir logs/walker/wpoprox03 --use-proper-time-limits --beta 0.2 --prox-target 0.02

python main.py --env-name "Walker2d-v2" --algo ppo --lr 3e-4 --num-steps 1000 --ppo-epoch 20 --num-processes 1 --num-env-steps 300000 --log-interval 1  --log-dir logs/walker/ppolonger --use-proper-time-limits --clip-param 0.2

#python main.py --env-name "Reacher-v2" --algo wpo --lr 1e-4 --num-steps 1000 --ppo-epoch 10 --num-processes 1 --num-env-steps 500000 --log-interval 1  --log-dir logs/reacher/wpo-ablated --use-proper-time-limits --beta 0.2 --no-wasserstein
