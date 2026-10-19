"""Behavioral checks for environment semantics, updates, and reproducibility."""
import unittest
from dataclasses import replace
import numpy as np
from q_learning import Config, make_env, q_update, epsilon_at, choose_action, evaluate


class TestQLearning(unittest.TestCase):
    def test_start_and_cliff_are_not_terminal(self):
        env = make_env(Config())
        self.assertEqual(env.reset(seed=8)[0], 36)
        state, reward, terminal, truncated, _ = env.step(1)
        self.assertEqual((state, reward, terminal, truncated), (36, -100, False, False))
        env.close()

    def test_shortest_safe_route(self):
        env = make_env(Config())
        env.reset(seed=8)
        total = 0
        for action in [0] + [1] * 11 + [2]:
            state, reward, terminal, truncated, _ = env.step(action)
            total += reward
        self.assertEqual((state, total, terminal, truncated), (47, -13, True, False))
        env.close()

    def test_wall(self):
        env = make_env(Config())
        env.reset(seed=8)
        self.assertEqual(env.step(3)[:4], (36, -1, False, False))
        env.close()

    def test_terminal_target(self):
        q = np.zeros((48, 4)); q[47] = 1000
        q_update(q, 35, 2, -1, 47, True, 0.5, 0.99)
        self.assertEqual(q[35, 2], -0.5)

    def test_time_limit_still_bootstraps(self):
        env = make_env(replace(Config(), max_steps=1))
        env.reset(seed=8)
        state, reward, terminated, truncated, _ = env.step(0)
        self.assertTrue(truncated); self.assertFalse(terminated)
        q = np.zeros((48, 4)); q[state] = 4
        q_update(q, 36, 0, reward, state, terminated, 0.5, 0.9)
        self.assertAlmostEqual(q[36, 0], 1.3)
        env.close()

    def test_noise_replaces_with_a_different_action(self):
        env = make_env(replace(Config(), noise=1.0))
        env.reset(seed=12)
        for _ in range(300):
            _, _, terminal, truncated, info = env.step(0)
            self.assertIn(info['executed_action'], [1, 2, 3])
            if terminal or truncated: env.reset()
        env.close()

    def test_noise_rate(self):
        env = make_env(replace(Config(), noise=0.25))
        env.reset(seed=12)
        count = 0
        for _ in range(20000):
            _, _, terminal, truncated, info = env.step(0)
            count += info['action_replaced']
            if terminal or truncated: env.reset()
        self.assertLess(abs(count / 20000 - 0.25), 0.015)
        env.close()

    def test_epsilon_schedule_and_seed_reproducibility(self):
        c = Config()
        self.assertEqual(epsilon_at(c, 0), 1.0)
        self.assertEqual(epsilon_at(c, 4999), 0.05)
        q = np.zeros((48, 4))
        a = np.random.default_rng(14); b = np.random.default_rng(14)
        self.assertEqual([choose_action(q, 0, .2, a) for _ in range(100)],
                         [choose_action(q, 0, .2, b) for _ in range(100)])

    def test_frozen_greedy_evaluation(self):
        q = np.zeros((48, 4)); q[36, 0] = 1
        q[24:35, 1] = 1; q[35, 2] = 1
        old = q.copy()
        data = evaluate(q, Config(), 30, 40000, 0, "test")
        np.testing.assert_array_equal(old, q)
        self.assertTrue((data.reward == -13).all())
        self.assertTrue((data.steps == 13).all())
        self.assertTrue((data.success == 1).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
