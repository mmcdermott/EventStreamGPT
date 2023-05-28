import sys

sys.path.append("../..")

import copy
import unittest
from unittest.mock import MagicMock, Mock, PropertyMock, call, patch

import torch

from EventStream.data.types import PytorchBatch
from EventStream.transformer.config import StructuredEventProcessingMode
from EventStream.transformer.generation.generation_utils import (
    StructuredGenerationMixin,
)

from ...utils import ConfigComparisonsMixin


class TestGenerationUtils(ConfigComparisonsMixin, unittest.TestCase):
    def test_expand_inputs_for_generation(self):
        batch = PytorchBatch(
            event_mask=torch.BoolTensor([[True, True, True, True], [False, True, True, True]]),
            time_delta=torch.FloatTensor([[0, 2, 5, 3], [0, 3, 2, 3]]),
            start_time=torch.FloatTensor([1.0, 1412.0]),
            static_indices=torch.LongTensor([[1, 2, 3], [1, 3, 0]]),
            static_measurement_indices=torch.LongTensor([[1, 2, 3], [1, 3, 0]]),
            dynamic_values_mask=torch.BoolTensor(
                [
                    [
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                        [False, False, False, True, True, True],
                        [False, False, False, False, True, True],
                    ],
                    [
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, True],
                        [False, False, False, False, True, True],
                    ],
                ]
            ),
            dynamic_measurement_indices=torch.LongTensor(
                [
                    [
                        [1, 0, 0, 0, 0, 0],
                        [1, 2, 0, 0, 0, 0],
                        [1, 2, 2, 3, 3, 3],
                        [1, 2, 2, 2, 3, 3],
                    ],
                    [
                        [1, 0, 0, 0, 0, 0],
                        [1, 2, 0, 0, 0, 0],
                        [1, 2, 2, 2, 2, 3],
                        [1, 2, 2, 2, 3, 3],
                    ],
                ]
            ),
            dynamic_indices=torch.LongTensor(
                [
                    [
                        [1, 0, 0, 0, 0, 0],
                        [2, 5, 0, 0, 0, 0],
                        [2, 4, 5, 7, 8, 9],
                        [2, 4, 5, 5, 8, 9],
                    ],
                    [
                        [1, 0, 0, 0, 0, 0],
                        [2, 5, 0, 0, 0, 0],
                        [2, 4, 5, 4, 4, 9],
                        [2, 4, 5, 5, 8, 9],
                    ],
                ]
            ),
            dynamic_values=torch.Tensor(
                [
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1.1, -1.1, 0.0],
                        [0, 0, 0, 0, -3.1, 0.2],
                    ],
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1.4],
                        [0, 0, 0, 0, -3.0, 1.2],
                    ],
                ]
            ),
        )

        want_expanded_batch = PytorchBatch(
            event_mask=torch.BoolTensor(
                [
                    [True, True, True, True],
                    [True, True, True, True],
                    [False, True, True, True],
                    [False, True, True, True],
                ]
            ),
            time_delta=torch.FloatTensor([[0, 2, 5, 3], [0, 2, 5, 3], [0, 3, 2, 3], [0, 3, 2, 3]]),
            start_time=torch.FloatTensor([1.0, 1.0, 1412.0, 1412.0]),
            static_indices=torch.LongTensor([[1, 2, 3], [1, 2, 3], [1, 3, 0], [1, 3, 0]]),
            static_measurement_indices=torch.LongTensor(
                [[1, 2, 3], [1, 2, 3], [1, 3, 0], [1, 3, 0]]
            ),
            dynamic_values_mask=torch.BoolTensor(
                [
                    [
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                        [False, False, False, True, True, True],
                        [False, False, False, False, True, True],
                    ],
                    [
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                        [False, False, False, True, True, True],
                        [False, False, False, False, True, True],
                    ],
                    [
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, True],
                        [False, False, False, False, True, True],
                    ],
                    [
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, True],
                        [False, False, False, False, True, True],
                    ],
                ]
            ),
            dynamic_measurement_indices=torch.LongTensor(
                [
                    [
                        [1, 0, 0, 0, 0, 0],
                        [1, 2, 0, 0, 0, 0],
                        [1, 2, 2, 3, 3, 3],
                        [1, 2, 2, 2, 3, 3],
                    ],
                    [
                        [1, 0, 0, 0, 0, 0],
                        [1, 2, 0, 0, 0, 0],
                        [1, 2, 2, 3, 3, 3],
                        [1, 2, 2, 2, 3, 3],
                    ],
                    [
                        [1, 0, 0, 0, 0, 0],
                        [1, 2, 0, 0, 0, 0],
                        [1, 2, 2, 2, 2, 3],
                        [1, 2, 2, 2, 3, 3],
                    ],
                    [
                        [1, 0, 0, 0, 0, 0],
                        [1, 2, 0, 0, 0, 0],
                        [1, 2, 2, 2, 2, 3],
                        [1, 2, 2, 2, 3, 3],
                    ],
                ]
            ),
            dynamic_indices=torch.LongTensor(
                [
                    [
                        [1, 0, 0, 0, 0, 0],
                        [2, 5, 0, 0, 0, 0],
                        [2, 4, 5, 7, 8, 9],
                        [2, 4, 5, 5, 8, 9],
                    ],
                    [
                        [1, 0, 0, 0, 0, 0],
                        [2, 5, 0, 0, 0, 0],
                        [2, 4, 5, 7, 8, 9],
                        [2, 4, 5, 5, 8, 9],
                    ],
                    [
                        [1, 0, 0, 0, 0, 0],
                        [2, 5, 0, 0, 0, 0],
                        [2, 4, 5, 4, 4, 9],
                        [2, 4, 5, 5, 8, 9],
                    ],
                    [
                        [1, 0, 0, 0, 0, 0],
                        [2, 5, 0, 0, 0, 0],
                        [2, 4, 5, 4, 4, 9],
                        [2, 4, 5, 5, 8, 9],
                    ],
                ]
            ),
            dynamic_values=torch.Tensor(
                [
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1.1, -1.1, 0.0],
                        [0, 0, 0, 0, -3.1, 0.2],
                    ],
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1.1, -1.1, 0.0],
                        [0, 0, 0, 0, -3.1, 0.2],
                    ],
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1.4],
                        [0, 0, 0, 0, -3.0, 1.2],
                    ],
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1.4],
                        [0, 0, 0, 0, -3.0, 1.2],
                    ],
                ]
            ),
        )

        got = StructuredGenerationMixin._expand_inputs_for_generation(batch, 2)
        self.assertEqual(want_expanded_batch, got)

        bad_batch = copy.deepcopy(batch)
        bad_batch.dynamic_values = None
        with self.assertRaises(TypeError):
            StructuredGenerationMixin._expand_inputs_for_generation(bad_batch, 2)

    def test_update_model_kwargs_for_generation(self):
        self.assertEqual(
            {"past": None},
            StructuredGenerationMixin._update_model_kwargs_for_generation(
                {"wrong_key": "present"}, {}
            ),
        )
        self.assertEqual(
            {"past": "present"},
            StructuredGenerationMixin._update_model_kwargs_for_generation(
                {"past_key_values": "present"}, {}
            ),
        )

    def test_get_stopping_criteria_max_length(self):
        cases = [
            {
                "msg": "Should work with max length",
                "kwargs": {"max_length": 10, "stopping_criteria": "foo"},
                "want_max_length_calls": [call(max_length=10)],
            },
            {
                "msg": "Should work with max length",
                "kwargs": {"max_length": None, "stopping_criteria": "bar"},
                "want_max_length_calls": [],
            },
        ]

        for i, case in enumerate(cases):
            with (
                patch(
                    "EventStream.transformer.generation.generation_utils.MaxLengthCriteria"
                ) as MockMaxLengthCriteria,
                patch(
                    "EventStream.transformer.generation.generation_utils.StoppingCriteriaList"
                ) as MockStoppingCriteriaList,
                self.subTest(f"Sub-test {i}: {case['msg']}"),
            ):
                M = StructuredGenerationMixin()
                M._merge_criteria_processor_list = Mock()

                input_kwargs = case.get("kwargs", {})
                M._get_stopping_criteria(**input_kwargs)

                MockStoppingCriteriaList.assert_called_once_with()

                want_max_length_calls = case.get("want_max_length_calls", [])
                self.assertNestedCalledWith(MockMaxLengthCriteria, want_max_length_calls)

                if want_max_length_calls:
                    mock_criteria = MockMaxLengthCriteria.return_value
                    MockStoppingCriteriaList.return_value.append.assert_called_once_with(
                        mock_criteria
                    )

                mock_criteria_list = MockStoppingCriteriaList.return_value

                M._merge_criteria_processor_list.assert_called_once_with(
                    mock_criteria_list, input_kwargs.get("stopping_criteria", None)
                )

    def test_merge_criteria_processor_list(self):
        M = StructuredGenerationMixin()
        self.assertEqual([1, 2, 3], M._merge_criteria_processor_list([1, 2, 3], []))
        self.assertEqual([1, 3, 3, 4, 4], M._merge_criteria_processor_list([1, 3], [3, 4, 4]))

    def test_generate(self):
        default_config_params = {
            "num_return_sequences": "config",
            "output_scores": "config",
            "output_attentions": "config",
            "output_hidden_states": "config",
            "return_dict_in_generate": "config",
            "max_length": 10,
            "max_seq_len": 20,
            "structured_event_processing_mode": StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT,
        }
        cases = [
            {
                "msg": "Should error if do_sample is False",
                "kwargs": {"do_sample": False},
                "should_raise": ValueError,
            },
            {
                "msg": "Should error if max length invalid",
                "kwargs": {"max_length": 25},
                "config_params": {"max_seq_len": 6},
                "should_raise": ValueError,
            },
            {
                "msg": "Should use passed kwargs if specified and config kwargs otherwise",
                "kwargs": {
                    "num_return_sequences": "kwargs",
                    "output_scores": "kwargs",
                    "output_attentions": "kwargs",
                    "output_hidden_states": "kwargs",
                    "return_dict_in_generate": "kwargs",
                    "max_length": 15,
                },
                "want_max_length": 15,
                "want_expand_size": "kwargs",
                "want_model_kwargs": {
                    "output_scores": "kwargs",
                    "output_attentions": "kwargs",
                    "output_hidden_states": "kwargs",
                    "return_dict_in_generate": "kwargs",
                    "use_cache": None,
                },
                "want_call_sample_CI": True,
            },
            {
                "msg": "Should selectively use passed kwargs if specified and config kwargs otherwise",
                "kwargs": {
                    "output_attentions": "kwargs",
                },
                "want_model_kwargs": {
                    "output_scores": "config",
                    "output_attentions": "kwargs",
                    "output_hidden_states": "config",
                    "return_dict_in_generate": "config",
                    "use_cache": None,
                },
                "want_call_sample_CI": True,
            },
            {
                "msg": "Should reflect use_cache passed by kwargs",
                "kwargs": {"use_cache": True},
                "want_model_kwargs": {
                    "output_scores": "config",
                    "output_attentions": "config",
                    "output_hidden_states": "config",
                    "return_dict_in_generate": "config",
                    "use_cache": True,
                },
                "want_call_sample_CI": True,
            },
            {
                "msg": "Should Error if passed an invalid structured event processing mode.",
                "config_params": {"structured_event_processing_mode": "foobar"},
                "should_raise": ValueError,
            },
            {
                "msg": "Should use NA if indicated",
                "config_params": {
                    "structured_event_processing_mode": StructuredEventProcessingMode.NESTED_ATTENTION
                },
                "want_model_kwargs": {
                    "output_scores": "config",
                    "output_attentions": "config",
                    "output_hidden_states": "config",
                    "return_dict_in_generate": "config",
                    "use_cache": None,
                },
                "want_call_sample_NA": True,
            },
            {
                "msg": "Should error if passed both max new events and max length.",
                "kwargs": {"max_length": 4, "max_new_events": 10},
                "should_raise": ValueError,
            },
            {
                "msg": "Should respect max_new_events.",
                "kwargs": {"max_new_events": 11},
                "input_seq_length": 5,
                "want_max_length": 16,
                "want_model_kwargs": {
                    "output_scores": "config",
                    "output_attentions": "config",
                    "output_hidden_states": "config",
                    "return_dict_in_generate": "config",
                    "use_cache": None,
                },
                "want_call_sample_CI": True,
            },
        ]

        for i, case in enumerate(cases):
            with patch(
                "EventStream.transformer.generation.generation_utils.torch.any"
            ) as torch_any_mock, self.subTest(f"Sub-test {i}: {case['msg']}"):
                M = StructuredGenerationMixin()
                M.config = Mock()

                batch = MagicMock()
                batch_seq_length = PropertyMock(return_value=case.get("input_seq_length", 5))
                type(batch).sequence_length = batch_seq_length

                config_params = copy.deepcopy(default_config_params)
                config_params.update(case.get("config_params", {}))
                for key, val in config_params.items():
                    setattr(M.config, key, val)

                M._sample_conditionally_independent = Mock()
                M._sample_nested_attention = Mock()
                M._expand_inputs_for_generation = Mock()
                M._get_stopping_criteria = Mock()

                torch_any_mock.return_value = False

                kwargs = {"do_sample": True}
                kwargs.update(case.get("kwargs", {}))

                if case.get("should_raise", None) is not None:
                    with self.assertRaises(case["should_raise"]):
                        M.generate(batch, **kwargs)
                else:
                    M.generate(batch, **kwargs)

                    torch_any_mock.assert_called_once_with(~batch["event_mask"][:, -1])

                    batch_seq_length.assert_called_once_with()

                    max_length = case.get("want_max_length", default_config_params["max_length"])
                    stopping_criteria_default = case.get("want_stopping_criteria", [])
                    M._get_stopping_criteria.assert_called_once_with(
                        max_length=max_length, stopping_criteria=stopping_criteria_default
                    )
                    stopping_criteria = M._get_stopping_criteria.return_value

                    num_return_sequences = case.get(
                        "want_expand_size", config_params.get("num_return_sequences", None)
                    )
                    M._expand_inputs_for_generation.assert_called_once_with(
                        batch, expand_size=num_return_sequences
                    )
                    batch = M._expand_inputs_for_generation.return_value

                    want_kwargs = {
                        "batch": batch,
                        "debug_seed": None,
                        "stopping_criteria": stopping_criteria,
                        **case.get("want_model_kwargs", {}),
                    }

                    if case.get("want_call_sample_CI", False):
                        M._sample_conditionally_independent.assert_called_once_with(**want_kwargs)
                    else:
                        M._sample_conditionally_independent.assert_not_called()

                    if case.get("want_call_sample_NA", False):
                        M._sample_nested_attention.assert_called_once_with(**want_kwargs)
                    else:
                        M._sample_nested_attention.assert_not_called()


if __name__ == "__main__":
    unittest.main()
