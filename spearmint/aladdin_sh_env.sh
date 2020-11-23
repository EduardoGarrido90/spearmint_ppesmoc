#!/bin/bash

#SBATCH -p hips,serial_requeue # Partition to submit to (comma separated)
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 15-00:00 # Runtime in D-HH:MM (or use minutes)
#SBATCH --mem 16000 # Memory in MB (see also --mem-per-cpu)
#SBATCH -o run_experiment.sh.output # File to which standard out will be written
#SBATCH -e run_experiment.sh.errors # File to which standard err will be written

export PROJECT_HOME=/home/eduardo/aladdin/pesmoc_aladdin/hardware_project/aladdin_plus_required_software
export LLVM_HOME=$PROJECT_HOME/llvm_3.4_compiled/
export TRACER_HOME=$PROJECT_HOME/LLVM-Tracer
export PATH=$LLVM_HOME/bin/:$PATH
export LD_LIBRARY_PATH=$LLVM_HOME/lib/:$LD_LIBRARY_PATH

export ALADDIN_HOME=$PROJECT_HOME/ALADDIN/
export BOOST_ROOT=$PROJECT_HOME/boost_1_55/boost_1_55_0
export LD_LIBRARY_PATH=$BOOST_ROOT/stage/lib:$LD_LIBRARY_PATH

export PYTHONPATH=$PYTHONPATH:$PROJECT_HOME/deepnet/
export PYTHONPATH=$PYTHONPATH:$PROJECT_HOME/deepnet/deepnet
export CUDA_BIN=/usr/local/cuda-8.0/bin/
export CUDA_LIB=/usr/local/cuda-8.0/lib64/
export PATH=${CUDA_BIN}:$PATH
export LD_LIBRARY_PATH=${CUDA_LIB}:$LD_LIBRARY_PATH
export USE_GPU=no

index=`cat ../simulation_index.txt`
method=`pwd  | tr "/" "\n" | tail -1`
database_folder=$PROJECT_HOME/garbage_${method}_simulation_${index}_mongodb
if [ ! -d "$database_folder" ]; then
    mkdir $database_folder
fi

port=`~/bin/python get_free_port.py`
echo $port
#Uncomment this line in the cluster.
#module load hpc/mongodb-2.2.2
mongod --fork --logpath $database_folder/log.log --dbpath $database_folder/ --port $port
export SPEARMINT_DB_ADDRESS="mongodb://localhost:$port/"
echo $SPEARMINT_DB_ADDRESS
#Change this line in the cluster.
python /home/eduardo/dev/python/spearmint/spearmint/main.py .
python evaluate_hypervolume.py .
mongod --shutdown --logpath $database_folder/log.log --dbpath $database_folder/ --port $port
