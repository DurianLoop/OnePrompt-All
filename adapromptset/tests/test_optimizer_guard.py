import unittest

import torch

from adapromptset.training.optimizer_guard import (
    FrozenParameterViolation,
    assert_optimizer_excludes_frozen_modules,
    freeze_module_parameters,
)


class OptimizerGuardTests(unittest.TestCase):
    def test_allows_optimizer_with_policy_parameters_only(self) -> None:
        matcher = torch.nn.Linear(2, 2)
        segmenter = torch.nn.Linear(2, 2)
        policy = torch.nn.Linear(2, 1)
        freeze_module_parameters(matcher)
        freeze_module_parameters(segmenter)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)

        report = assert_optimizer_excludes_frozen_modules(
            optimizer,
            frozen_modules={"matcher": matcher, "segmenter": segmenter},
        )

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["optimizer_parameter_count"], sum(1 for _ in policy.parameters()))
        self.assertEqual(report["frozen_parameter_count"], 4)
        self.assertTrue(all(not parameter.requires_grad for parameter in matcher.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in segmenter.parameters()))

    def test_rejects_optimizer_that_contains_matcher_parameter(self) -> None:
        matcher = torch.nn.Linear(2, 2)
        policy = torch.nn.Linear(2, 1)
        freeze_module_parameters(matcher)
        optimizer = torch.optim.SGD(
            [
                {"params": policy.parameters()},
                {"params": matcher.parameters()},
            ],
            lr=1e-3,
        )

        with self.assertRaises(FrozenParameterViolation) as context:
            assert_optimizer_excludes_frozen_modules(optimizer, frozen_modules={"matcher": matcher})

        message = str(context.exception)
        self.assertIn("matcher", message)
        self.assertIn("optimizer", message)

    def test_rejects_optimizer_that_contains_segmenter_parameter_even_if_requires_grad_false(self) -> None:
        segmenter = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Linear(4, 1))
        freeze_module_parameters(segmenter)
        optimizer = torch.optim.AdamW(segmenter.parameters(), lr=1e-3)

        with self.assertRaises(FrozenParameterViolation):
            assert_optimizer_excludes_frozen_modules(optimizer, frozen_modules={"segmenter": segmenter})


if __name__ == "__main__":
    unittest.main()
