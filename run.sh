set -e

paths=(
'base'

'network/RNN'
'network/LSTM'

# 'online'

'features'

'output/sigmoid'
'output/tanh'

'loss/binary'
'loss/multiclass'

'hyperparameters/window/10'
'hyperparameters/window/15'
)

if [ ! -z $1 ]; then
	paths=($1)
fi

for path in "${paths[@]}"
do
	save_dir="exp/$path"
	echo $save_dir
    echo
    python -u main.py $save_dir | tee $save_dir/log
    echo '============'
done
