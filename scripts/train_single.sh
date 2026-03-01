#!/bin/bash -l
#SBATCH --time=02:30:00
#SBATCH --nodes=1 --cpus-per-task=1 --ntasks=1
#SBATCH --partition=med
#SBATCH --mem=5G

#SBATCH --mail-user=timon.sachweh@tu-dortmund.de
#SBATCH --mail-type=ALL
#SBATCH --signal=15@30

#SBATCH --output=/work/smtisach/logs/rlpn-%j.out
#SBATCH --error=/work/smtisach/logs/rlpn-%j.out

export USER=smtisach
export ALGO_DIR=/work/${USER}/reinforcementLearningPetriNets

module purge
module load python/3.9.10-wip

echo "sbatch: START SLURM_JOB_ID $SLURM_JOB_ID (SLURM_TASK_PID $SLURM_TASK_PID) on $SLURMD_NODENAME"
echo "sbatch: SLURM_JOB_NODELIST $SLURM_JOB_NODELIST"
echo "sbatch: SLURM_JOB_ACCOUNT $SLURM_JOB_ACCOUNT"

et=$1
efe=$2
ls=$3
rt=$4
reward="dynamic_reward"
ms=$5
mc=$6
mw=$7
mmw=$8
mt=$9

cd ${ALGO_DIR}

echo "train.py --train --exploration_timesteps $et --exploration-final-epsilon $efe \
               --learning-starts $ls --random-timesteps $rt --reward-function $reward \
               --m-success $ms --m-cars-driven $mc --m-waiting-time $mw \
               --m-max-waiting-time $mmw --m-timestep $mt"
python3 train.py --train --exploration-timesteps $et --exploration-final-epsilon $efe \
                 --learning-starts $ls --random-timesteps $rt --reward-function $reward \
                 --m-success $ms --m-cars-driven $mc --m-waiting-time $mw \
                 --m-max-waiting-time $mmw --m-timestep $mt