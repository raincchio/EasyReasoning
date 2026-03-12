cd /vepfs-dev/xing/workspace/EasyReasoning/
export PYTHONPATH=$(pwd):$(pwd)
python=/vepfs-dev/xing/miniconda3/envs/test/bin/python

declare -i i=7
available_gpus=(0 1 2 3 4 5 6 7)
cycle=4
for h in 1
do
    for l in 2
    do
    #      echo ${wd}
      export CUDA_VISIBLE_DEVICES=${available_gpus[$((i % ${#available_gpus[@]}))]}
      i+=1
      cmd="nohup ${python} -um urm-h-best --num_layers 4 --hidden_size 512 --H_cycles ${h} --L_cycles ${l}  --cycle_per_data ${cycle} >>/dev/null 2>&1 &"
      eval $cmd
    done
done

#for h in 1
#do
#    for l in 2 3
#    do
#    #      echo ${wd}
#      export CUDA_VISIBLE_DEVICES=${available_gpus[$((i % ${#available_gpus[@]}))]}
#      i+=1
#      cmd="nohup ${python} -um urm-h-best --num_layers 4 --hidden_size 512 --H_cycles ${h} --L_cycles ${l}  --cycle_per_data ${cycle} >>/dev/null 2>&1 &"
#      eval $cmd
#    done
#done
#
#for h in 2
#do
#    for l in 1 2 3
#    do
#    #      echo ${wd}
#      export CUDA_VISIBLE_DEVICES=${available_gpus[$((i % ${#available_gpus[@]}))]}
#      i+=1
#      cmd="nohup ${python} -um urm-h-best --num_layers 4 --hidden_size 512 --H_cycles ${h} --L_cycles ${l}  --cycle_per_data ${cycle} >>/dev/null 2>&1 &"
#      eval $cmd
#    done
#done








#  export CUDA_VISIBLE_DEVICES=5
#  cmd="nohup ${python} -um urm-h-best --hidden_size 512 --num_layers 4 --H_cycles 1 --L_cycles 7  --cycle_per_data 4 >>/dev/null 2>&1 &"
#  eval $cmd
#
#  export CUDA_VISIBLE_DEVICES=2
#  cmd="nohup ${python} -um urm-h-best --hidden_size 512 --num_layers 4 --H_cycles 3 --L_cycles 1  --cycle_per_data 8  >>/dev/null 2>&1 &"
#  eval $cmd
#  export CUDA_VISIBLE_DEVICES=6
#  cmd="nohup ${python} -um urm-h-best --hidden_size 512 --num_layers 4 --H_cycles 3 --L_cycles 1  --cycle_per_data 16  >>/dev/null 2>&1 &"
#  eval $cmd

#  export CUDA_VISIBLE_DEVICES=3
#  cmd="nohup ${python} -um urm-h-best --hidden_size 512 --num_layers 4 --H_cycles 1 --L_cycles 5  --cycle_per_data ${cycle_per_data}  >>/dev/null 2>&1 &"
#  eval $cmd

#  export CUDA_VISIBLE_DEVICES=4
#  cmd="nohup ${python} -um urm-h-best --hidden_size 512 --num_layers 4 --H_cycles 0 --L_cycles 1  --cycle_per_data ${cycle_per_data} >>/dev/null 2>&1 &"
#  eval $cmd
#
#  export CUDA_VISIBLE_DEVICES=5
#  cmd="nohup ${python} -um urm-h-best --hidden_size 512 --num_layers 4 --H_cycles 0 --L_cycles 2  --cycle_per_data ${cycle_per_data}  >>/dev/null 2>&1 &"
#  eval $cmd
#  export CUDA_VISIBLE_DEVICES=6
#  cmd="nohup ${python} -um urm-h-best --hidden_size 512 --num_layers 4 --H_cycles 1 --L_cycles 2  --cycle_per_data ${cycle_per_data}  >>/dev/null 2>&1 &"
#  eval $cmd
#
#  export CUDA_VISIBLE_DEVICES=7
#  cmd="nohup ${python} -um urm-h-best --hidden_size 512 --num_layers 4 --H_cycles 1 --L_cycles 1  --cycle_per_data ${cycle_per_data}  >>/dev/null 2>&1 &"
#  eval $cmd



