#!/bin/bash

exploration_timesteps=(400000)
exploration_final_epsilon=("0.04")
learning_starts=(100000)
random_timesteps=(100000)
reward_function=("constraint_driven_waiting_times_timesteps" "constraint_avg_waiting_times_and_timesteps" "constraint_timestep" "driven_waiting_times_timesteps" "avg_waiting_times_and_timesteps" "timestep" "constraint_cars_driven_timestep" "cars_driven_timestep" "base_reward")

reward_function=("dynamic_reward")
m_success=("0.1" "1.5")
m_cars_driven=("0.1" "1.5")
m_waiting_time=("0.1" "1.5")
m_max_waiting_time=("0.1" "1.5")
m_timestep=("0.1" "1.5")


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
                      echo "train.py --train --exploration_timesteps $et --exploration-final-epsilon $efe \
                                     --learning-starts $ls --random-timesteps $rt --reward-function $reward \
                                     --m-success $ms --m-cars-driven $mc --m-waiting-time $mw \
                                     --m-max-waiting-time $mmw --m-timestep $mt"
                      python train.py --train --exploration-timesteps $et --exploration-final-epsilon $efe \
                                      --learning-starts $ls --random-timesteps $rt --reward-function $reward \
                                      --m-success $ms --m-cars-driven $mc --m-waiting-time $mw \
                                      --m-max-waiting-time $mmw --m-timestep $mt &
                    done
                  done
                done
              done
            done
          else
            echo "train.py --train --exploration_timesteps $et --exploration-final-epsilon $efe \
                           --learning-starts $ls --random-timesteps $rt --reward-function $reward"
            python train.py --train --exploration-timesteps $et --exploration-final-epsilon $efe --learning-starts $ls \
                            --random-timesteps $rt --reward-function $reward &
          fi
        done
      done
    done
  done
done