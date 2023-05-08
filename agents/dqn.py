from skrl.agents.torch.dqn import DQN
from skrl.utils.model_instantiators import deterministic_model, Shape
from agents import c_deterministic_model, CDQN


def get_dqn_model(env, memory, cfg, constrained=False):
    models_dqn = {}
    if not constrained:
        models_dqn["q_network"] = deterministic_model(observation_space=env.observation_space,
                                                  action_space=env.action_space,
                                                  device=env.device,
                                                  clip_actions=False,
                                                  input_shape=Shape.OBSERVATIONS,
                                                  hiddens=[64, 64],
                                                  hidden_activation=["relu", "relu"],
                                                  output_shape=Shape.ACTIONS,
                                                  output_activation=None,
                                                  output_scale=1.0)
        models_dqn["target_q_network"] = deterministic_model(observation_space=env.observation_space,
                                                         action_space=env.action_space,
                                                         device=env.device,
                                                         clip_actions=False,
                                                         input_shape=Shape.OBSERVATIONS,
                                                         hiddens=[64, 64],
                                                         hidden_activation=["relu", "relu"],
                                                         output_shape=Shape.ACTIONS,
                                                         output_activation=None,
                                                         output_scale=1.0)


    else:
        models_dqn["q_network"] = c_deterministic_model(observation_space=env.observation_space,
                                                  action_space=env.action_space,
                                                  device=env.device,
                                                  clip_actions=False,
                                                  input_shape=Shape.OBSERVATIONS,
                                                  hiddens=[64, 64],
                                                  hidden_activation=["relu", "relu"],
                                                  output_shape=Shape.ACTIONS,
                                                  output_activation=None,
                                                  output_scale=1.0)
        models_dqn["target_q_network"] = c_deterministic_model(observation_space=env.observation_space,
                                                         action_space=env.action_space,
                                                         device=env.device,
                                                         clip_actions=False,
                                                         input_shape=Shape.OBSERVATIONS,
                                                         hiddens=[64, 64],
                                                         hidden_activation=["relu", "relu"],
                                                         output_shape=Shape.ACTIONS,
                                                         output_activation=None,
                                                         output_scale=1.0)

    # Initialize the models' parameters (weights and biases) using a Gaussian distribution
    for model in models_dqn.values():
            model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

    agent = DQN(models=models_dqn, memory=memory, cfg=cfg, observation_space=env.observation_space,
                action_space=env.action_space, device=env.device) if constrained \
        else CDQN(models=models_dqn, memory=memory, cfg=cfg, observation_space=env.observation_space,
                action_space=env.action_space, device=env.device)
    return agent




