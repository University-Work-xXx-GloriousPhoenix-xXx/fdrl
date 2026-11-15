"""Клас MDP (наданий у демо КП1) для власного дискретного середовища."""
import numpy as np

class MDP:
    def __init__(self, P, initial_state=0, max_steps=100):
        self.P = P
        self.initial_state = initial_state
        self.max_steps = max_steps

        self.n_states = len(P)      # !!
        self.n_actions = len(P[0])  # !!

        self.observation_space = range(self.n_states)  # !!
        self.action_space = range(self.n_actions)      # !!

        self.rng = np.random.default_rng()

        self._validate_model()

        self.state = None
        self.steps = 0
        self.episode_over = True

    def _validate_model(self):
        """Перевіряє коректність моделі MDP."""

        if self.initial_state not in self.observation_space:
            raise ValueError(
                f"Некоректний початковий стан: {self.initial_state}"
            )

        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps має бути додатним числом")

        expected_states = set(self.observation_space)
        expected_actions = set(self.action_space)

        if set(self.P) != expected_states:
            raise ValueError(
                f"Стани мають бути пронумеровані як {expected_states}"
            )

        for state in self.observation_space:
            if set(self.P[state]) != expected_actions:
                raise ValueError(
                    f"Некоректний набір дій у стані {state}"
                )

            for action in self.action_space:
                transitions = self.P[state][action]

                if not transitions:
                    raise ValueError(
                        f"Немає переходів для стану {state}, дії {action}"
                    )

                probabilities = []

                for transition in transitions:
                    if len(transition) != 4:
                        raise ValueError(
                            "Перехід повинен мати формат "
                            "(probability, next_state, reward, terminated)"
                        )

                    probability, next_state, _, _ = transition

                    if probability < 0:
                        raise ValueError(
                            "Ймовірність не може бути від'ємною"
                        )

                    if next_state not in self.observation_space:
                        raise ValueError(
                            f"Некоректний наступний стан: {next_state}"
                        )

                    probabilities.append(probability)

                if not np.isclose(sum(probabilities), 1.0):
                    raise ValueError(
                        f"Сума ймовірностей для стану {state}, "
                        f"дії {action} не дорівнює 1"
                    )

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.state = self.initial_state
        self.steps = 0
        self.episode_over = False

        return self.state, {}

    def step(self, action):
        if self.episode_over:
            raise RuntimeError(
                "Епізод завершено. Викличте reset()."
            )

        if action not in self.action_space:
            raise ValueError(f"Некоректна дія: {action}")

        transitions = self.P[self.state][action]
        probabilities = [
            transition[0] for transition in transitions
        ]

        index = self.rng.choice(
            len(transitions),
            p=probabilities,
        )

        probability, next_state, reward, terminated = transitions[index]

        self.state = next_state
        self.steps += 1

        truncated = (
            not terminated
            and self.max_steps is not None
            and self.steps >= self.max_steps
        )

        self.episode_over = terminated or truncated

        info = {
            "prob": probability,
            "steps": self.steps,
        }

        return (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        )
