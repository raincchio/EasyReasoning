cd /vepfs-dev/xing/workspace/EasyReasoning/
export PYTHONPATH=$(pwd):$(pwd)
python=/vepfs-dev/xing/miniconda3/envs/test/bin/python

declare -i i=0
available_gpus=(0 2)


for hidden_size in 256
do
  for num_layers in 1 2
  do
#      echo ${wd}
    export CUDA_VISIBLE_DEVICES=${available_gpus[$((i % ${#available_gpus[@]}))]}
    i+=1

    cmd="nohup ${python} -um urm-h-best --hidden_size ${hidden_size} --intermediate_size ${hidden_size} --num_layers ${num_layers} >>/dev/null 2>&1 &"

#        misc="--enable_adam --enable_redo --redo_check_interval ${itv}"
#        cmd="nohup ${python} -um dqn --seed=${seed} --env_id=${env} ${misc} --exp_name  999_adam_redo_itv_${itv}  >>/dev/null 2>&1 &"
    echo $cmd
    eval $cmd
  done
done
