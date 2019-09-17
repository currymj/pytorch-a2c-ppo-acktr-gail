python main.py --env-name "Reacher-v2" --algo wpo --lr 1e-4 --num-steps 1000 --ppo-epoch 10 --num-processes 8 --num-env-steps 1000000 --log-interval 1  --log-dir logs/reacher/wpo --use-proper-time-limits --beta 0.2

python main.py --env-name "Reacher-v2" --algo ppo --lr 1e-4 --num-steps 1000 --clip-param 0.3 --ppo-epoch 10 --num-processes 8 --num-env-steps 1000000 --log-interval 1  --log-dir logs/reacher/ppo_wide --use-proper-time-limits
