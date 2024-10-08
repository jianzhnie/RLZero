import multiprocessing as mp


def run_actor(actor_id, env_name, model, replay_buffer):
    actor = Actor(actor_id, env_name, model, replay_buffer)
    actor.run()


def main():
    env_name = "CartPole-v1"
    state_dim = 4
    action_dim = 2
    num_actors = 4
    batch_size = 32
    capacity = 10000
    alpha = 0.6
    beta = 0.4

    model = QNetwork(state_dim, action_dim)
    target_model = QNetwork(state_dim, action_dim)
    target_model.load_state_dict(model.state_dict())
    replay_buffer = PrioritizedReplayBuffer(capacity, alpha, beta)

    processes = []
    for actor_id in range(num_actors):
        p = mp.Process(
            target=run_actor, args=(actor_id, env_name, model, replay_buffer)
        )
        p.start()
        processes.append(p)

    learner = Learner(model, target_model, replay_buffer, batch_size)

    for _ in range(1000):
        learner.train()
        learner.update_target_model()

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
