#!/bin/bash -l


export USER=smtisach
export ALGO_DIR=/home/${USER}/scripts

cd ${ALGO_DIR}
exploration_timesteps=(400000)
exploration_final_epsilon=("0.04")
learning_starts=(100000)
random_timesteps=(100000)

reward_function=("dynamic_reward")
#m_success=("0.1" "1.0" "1.5")
#m_cars_driven=("0.1" "1.5")
#m_waiting_time=("0.1" "1.5")
#m_max_waiting_time=("0.1" "1.5")
#m_timestep=("0.1" "1.5")
m_success=("0.0" "1.0" "1.5")
m_cars_driven=("0.0" "1.0" "1.5")
m_waiting_time=("0.0" "1.0" "1.5")
m_max_waiting_time=("0.0" "1.0" "1.5")
m_timestep=("0.0" "1.0" "1.5")


#m_success=("1.0")
#m_cars_driven=("1.0")
#m_waiting_time=("1.0")
#m_max_waiting_time=("1.0")
#m_timestep=("1.0")

for reward in "${reward_function[@]}"
do
  for rt in "${random_timesteps[@]}"
  do
    for ls in "${learning_starts[@]}"
    do
      for efe in "${exploration_final_epsilon[@]}"
      do
        for et in "${exploration_timesteps[@]}"
        do
          if [ "$reward" = "dynamic_reward" ]
          then
            for ms in "${m_success[@]}"
            do
              for mc in "${m_cars_driven[@]}"
              do
                for mw in "${m_waiting_time[@]}"
                do
                  for mmw in "${m_max_waiting_time[@]}"
                  do
                    for mt in "${m_timestep[@]}"
                    do
                      sbatch train_single.sh $et $efe $ls $rt $ms $mc $mw $mmw $mt
                    done
                  done
                done
              done
            done
          else
            echo "train.py --train --exploration_timesteps $et --exploration-final-epsilon $efe \
                           --learning-starts $ls --random-timesteps $rt --reward-function $reward"
            sbatch train_single.sh $et $efe $ls $rt $reward
          fi
        done
      done
    done
  done
done