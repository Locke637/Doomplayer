#!/usr/bin/env bash
BASEDIR=$(dirname "$( cd "$( dirname "$0" )" && pwd )")
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [train|resume|test]"
    exit 1
fi

COMMAND="$1"
MODEL=ppo
CHECK_POINT=$BASEDIR/checkpoints/deathmatch__ppo_cp.pth
CONFIG=$BASEDIR/environments/deathmatch.cfg
INSTANCE=deathmatch

if [ $COMMAND == 'train' ]
then
    python $BASEDIR/src/main.py \
    --mode train \
    --episode_size 64 \
    --batch_size 5 \
    --episode_discount 0.95 \
    --model $MODEL \
    --doom_instance $INSTANCE \
    --vizdoom_config $CONFIG \
    --skiprate 4 \
    --frame_num 1 \
    --checkpoint_file $CHECK_POINT \
    --checkpoint_rate 100 \
    --episode_num 50000
elif [ $COMMAND == 'resume' ]
then
    python $BASEDIR/src/main.py \
    --mode train \
    --episode_size 64 \
    --batch_size 5 \
    --episode_discount 0.95 \
    --model $MODEL \
    --load $CHECK_POINT \
    --doom_instance $INSTANCE \
    --vizdoom_config $CONFIG \
    --skiprate 4 \
    --frame_num 1 \
    --checkpoint_file $CHECK_POINT \
    --checkpoint_rate 100 \
    --episode_num 50000
elif [ $COMMAND == 'test' ]
then
    python $BASEDIR/src/main.py \
    --mode test \
    --batch_size 1 \
    --model $MODEL \
    --load $CHECK_POINT \
    --doom_instance $INSTANCE \
    --vizdoom_config $CONFIG \
    --skiprate 1 \
    --frame_num 1
else
    echo "'$COMMAND' is unknown command."
fi