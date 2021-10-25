from gym.envs.registration import register

register(
    id='adv-carla-v0',
    entry_point='adv_carla.envs:AdversarialCARLAEnv',
)
