import numpy as np


'''
def agent(pi, s):
def policy_deterministic_to_stochastic (det_policy, nA, epsilon=0.0):

'''



def agent(pi, s):
    """
    Генерує дію на основі стохастичної політики та поточного стану
    Generates action based on stochastic policy and current state
    
    Parameters:
    - pi: policy, стохастична політика [nS, nA] - матриця ймовірностей дій для кожного стану
    - s:  state,  поточний стан (int)
    
    Returns:
    - a:  action, обрана дія (int)
    """
    # Get probability distribution of actions for given state
    action_probabilities = pi[s]
    
    # Validate probability distribution
    #if not np.isclose(np.sum(action_probabilities), 1.0):
    #    print(f"Увага: ймовірності для стану {state} не сумуються до 1 (сума = {np.sum(action_probabilities):.6f})")
    #    # Нормалізуємо ймовірності / Normalize probabilities
    #    action_probabilities = action_probabilities / np.sum(action_probabilities)
    
    # Choose action randomly according to probability distribution
    a = np.random.choice(len(action_probabilities), p=action_probabilities)
    
    return a


#
def policy_deterministic_to_stochastic (det_policy, nA: int, epsilon: float = 0.0):
    """
    Перетворює детерміновану політику на стохастичну
    метод перетворення 'epsilon_greedy'
    Converts deterministic policy to stochastic
    
    Parameters:
    - det_policy: детермінована політика [nS] - дія для кожного стану
    - nA: Number of actions
    - epsilon: параметр для epsilon-жадібного методу
    
    Returns:
    - stochastic_policy: стохастична політика [nS, nA] - ймовірності дій для кожного стану
    """

    nS = len(det_policy)  #  Number of states
    stochastic_policy = np.zeros((nS, nA))

    # Epsilon-greedy method: main probability mass on best action
    for s in range(nS):
        best_action = det_policy[s]
        for a in range(nA):
            if a == best_action:
                stochastic_policy[s, a] = 1 - epsilon + epsilon/nA
            else:
                stochastic_policy[s, a] = epsilon/nA
   
    return stochastic_policy
    

def run_episode(env, policy, gamma, max_steps=None):
    state, _ = env.reset()
    discounted_return = 0.0
    trajectory = []
    steps = 0

    while True:
        action = agent(policy, state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        discounted_return += gamma**steps * reward
        steps += 1

        if max_steps is not None and steps >= max_steps and not terminated:
            truncated = True

        trajectory.append((state, action, reward, next_state, terminated, truncated))

        if terminated or truncated:
            break
        state = next_state

    return discounted_return, steps, trajectory



def compute_transition_matrix(P, policy):
    policy = np.asarray(policy, dtype=float)
    nS, nA = len(P), len(P[0])

    if policy.shape != (nS, nA):
        raise ValueError(f"Очікувана форма policy: ({nS}, {nA})")
    if np.any(policy < 0) or not np.allclose(policy.sum(axis=1), 1):
        raise ValueError("Кожен рядок policy має задавати розподіл імовірностей.")

    T = np.zeros((nS, nS))
    for s in range(nS):
        for a in range(nA):
            for probability, next_state, _, _ in P[s][a]:
                T[s, next_state] += policy[s, a] * probability

    return T
    

def validate_transition_matrix(T, tol=1e-10):
    T = np.asarray(T, dtype=float)

    return bool(
        T.ndim == 2
        and T.shape[0] > 0
        and T.shape[0] == T.shape[1]
        and np.all(np.isfinite(T))
        and np.all(T >= -tol)
        and np.allclose(T.sum(axis=1), 1.0, atol=tol, rtol=0)
    )


def evolve_distribution(mu0, T, theta=1e-8, max_iter=500):
    T = np.asarray(T, dtype=float)
    mu = np.asarray(mu0, dtype=float).copy()

    if not validate_transition_matrix(T):
        raise ValueError("Некоректна матриця переходів T.")
    if mu.shape != (T.shape[0],) or np.any(mu < 0) or not np.isclose(mu.sum(), 1):
        raise ValueError("mu0 має бути розподілом імовірностей за станами.")

    for n_iter in range(1, max_iter + 1):
        mu_next = mu @ T
        if np.max(np.abs(mu_next - mu)) < theta:
            return mu_next, n_iter
        mu = mu_next

    raise RuntimeError(f"Розподіл не збігся за {max_iter} кроків.")


def compute_policy_reward_vector(P, policy):
    policy = np.asarray(policy, dtype=float)
    r_pi = np.zeros(len(P))

    for s in range(len(P)):
        for a in range(policy.shape[1]):
            for probability, next_state, reward, done in P[s][a]:
                r_pi[s] += policy[s, a] * probability * reward

    return r_pi
    

def policy_evaluation(P, policy, gamma, theta=1e-8, max_iter=10_000):
    v = np.zeros(len(P))
    policy = np.asarray(policy, dtype=float)

    for n_iter in range(1, max_iter + 1):
        v_new = np.zeros_like(v)

        for s in range(len(P)):
            for a, action_probability in enumerate(policy[s]):
                for probability, next_state, reward, done in P[s][a]:
                    continuation = 0 if done else gamma * v[next_state]
                    v_new[s] += action_probability * probability * (reward + continuation)

        if np.max(np.abs(v_new - v)) < theta:
            return v_new, n_iter
        v = v_new

    raise RuntimeError(f"Цінності не збіглися за {max_iter} ітерацій.")

