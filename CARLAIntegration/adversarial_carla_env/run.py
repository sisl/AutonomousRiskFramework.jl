from adversarial_carla_env import AdversarialCARLAEnv


if __name__ == "__main__":
    env = None
    try:
        env = AdversarialCARLAEnv()
        result = env.run()
    finally:
        if env is not None:
            print("Destroying scenario runner.")
            env.destroy()
            del env
    
