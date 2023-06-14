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
            static_measurement_indices=torch.LongTensor([[1, 2, 3], [1, 2, 3], [1, 3, 0], [1, 3, 0]]),
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
            StructuredGenerationMixin._update_model_kwargs_for_generation({"wrong_key": "present"}, {}),
        )
        self.assertEqual(
            {"past": "present"},
            StructuredGenerationMixin._update_model_kwargs_for_generation({"past_key_values": "present"}, {}),
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
                    MockStoppingCriteriaList.return_value.append.assert_called_once_with(mock_criteria)

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
            "return_dict_in_generate": False,
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
                    "max_length": 15,
                },
                "want_max_length": 15,
                "want_expand_size": "kwargs",
                "want_model_kwargs": {
                    "output_attentions": "kwargs",
                    "output_hidden_states": "kwargs",
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
                    "output_attentions": "kwargs",
                    "output_hidden_states": "config",
                    "use_cache": None,
                },
                "want_call_sample_CI": True,
            },
            {
                "msg": "Should reflect use_cache passed by kwargs",
                "kwargs": {"use_cache": True},
                "want_model_kwargs": {
                    "output_attentions": "config",
                    "output_hidden_states": "config",
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
                    "output_attentions": "config",
                    "output_hidden_states": "config",
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
                    "output_attentions": "config",
                    "output_hidden_states": "config",
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

                input_seq_length = case.get("input_seq_length", 5)
                batch = MagicMock()
                batch_seq_length = PropertyMock(return_value=input_seq_length)
                type(batch).sequence_length = batch_seq_length

                max_length = case.get("want_max_length", default_config_params["max_length"])
                want_n_events = max_length - input_seq_length

                config_params = copy.deepcopy(default_config_params)
                config_params.update(case.get("config_params", {}))
                for key, val in config_params.items():
                    setattr(M.config, key, val)

                CI_sample_mocks = [
                    (MagicMock(), MagicMock(), MagicMock(), MagicMock(), {"CI_event": i}) for i in range(100)
                ]
                M._conditionally_independent_sample_event = Mock(side_effect=CI_sample_mocks)

                NA_sample_mocks = [
                    (MagicMock(), MagicMock(), MagicMock(), MagicMock(), {"NA_event": i}) for i in range(100)
                ]
                M._nested_attention_sample_event = Mock(side_effect=NA_sample_mocks)

                M._expand_inputs_for_generation = MagicMock()

                stopping_criteria = Mock(side_effect=[False] * (want_n_events - 1) + [True])
                M._get_stopping_criteria = Mock(return_value=stopping_criteria)

                torch_any_mock.return_value = False

                kwargs = {"do_sample": True}
                kwargs.update(case.get("kwargs", {}))

                if case.get("should_raise", None) is not None:
                    with self.assertRaises(case["should_raise"]):
                        M.generate(batch, **kwargs)
                else:
                    got = M.generate(batch, **kwargs)

                    torch_any_mock.assert_called_once_with(~batch["event_mask"][:, -1])

                    batch_seq_length.assert_called_once_with()

                    stopping_criteria_default = case.get("want_stopping_criteria", [])
                    M._get_stopping_criteria.assert_called_once_with(
                        max_length=max_length, stopping_criteria=stopping_criteria_default
                    )

                    num_return_sequences = case.get(
                        "want_expand_size", config_params.get("num_return_sequences", None)
                    )
                    M._expand_inputs_for_generation.assert_called_once_with(
                        batch, expand_size=num_return_sequences
                    )
                    batch = M._expand_inputs_for_generation.return_value

                    model_kwargs = case.get("want_model_kwargs", {})
                    want_sample_calls = []
                    for event in range(want_n_events):
                        want_sample_calls.append(call(batch, event, **model_kwargs))

                        if case.get("want_call_sample_CI", False):
                            sample = CI_sample_mocks[event]
                        elif case.get("want_call_sample_NA", False):
                            sample = NA_sample_mocks[event]
                        else:
                            raise ValueError()

                        batch = sample[0]
                        model_kwargs = sample[4]

                    if case.get("want_call_sample_CI", False):
                        self.assertNestedCalledWith(
                            M._conditionally_independent_sample_event, want_sample_calls
                        )
                    else:
                        M._conditionally_independent_sample_event.assert_not_called()

                    if case.get("want_call_sample_NA", False):
                        self.assertNestedCalledWith(M._nested_attention_sample_event, want_sample_calls)
                    else:
                        M._nested_attention_sample_event.assert_not_called()

                    self.assertEqual(got, batch)

    def test_conditionally_independent_sample_event(self):
        class MockMixin(StructuredGenerationMixin, MagicMock):
            def __init__(self, *args, **kwargs):
                StructuredGenerationMixin.__init__(self)
                MagicMock.__init__(self, *args, **kwargs)

        M = MockMixin()
        M.prepare_inputs_for_generation = Mock(return_value={"prepared": True})
        M._update_model_kwargs_for_generation = Mock()

        batch = MagicMock()
        model_kwargs = {"foo": "bar"}

        got = M._conditionally_independent_sample_event(batch, 0, **model_kwargs)

        M.prepare_inputs_for_generation.assert_called_once_with(batch, **model_kwargs)
        M.assert_called_once_with(prepared=True, return_dict=True, is_generation=True)

        output = M.return_value

        M._update_model_kwargs_for_generation.assert_called_once_with(output, model_kwargs)
        out_kwargs = M._update_model_kwargs_for_generation.return_value

        output.preds.slice.assert_called_once_with((slice(None), -1))
        pred = output.preds.slice.return_value

        pred.sample.assert_called_once_with(batch.event_mask)
        sample = pred.sample.return_value

        sample.append_to_batch.assert_called_once_with(batch, M.config)
        batch = sample.append_to_batch.return_value

        sample.update_last_event_data.assert_called_once_with(batch, M.config)
        batch = sample.update_last_event_data.return_value

        self.assertNestedEqual((batch, pred, output.attentions, output.hidden_states, out_kwargs), got)

    def test_nested_attention_sample_event_first_event(self):
        class MockMixin(StructuredGenerationMixin, MagicMock):
            def __init__(self, *args, **kwargs):
                StructuredGenerationMixin.__init__(self)
                MagicMock.__init__(self, *args, **kwargs)

        model_outputs = [MagicMock() for _ in range(100)]
        M = MockMixin(side_effect=model_outputs)
        M.config.measurements_per_dep_graph_level = [[], ["foo", "bar"], ["baz"]]

        M.prepare_inputs_for_generation = Mock(side_effect=[{f"prepared_{i}": True} for i in range(10)])
        M._update_model_kwargs_for_generation = Mock(side_effect=[{"updated_{i}": True} for _ in range(10)])

        batch = MagicMock()
        model_kwargs = {"foo": "bar"}

        got = M._nested_attention_sample_event(batch, 0, **model_kwargs)

        want_measurements_to_fill = [{"time"}, ["foo", "bar"], ["baz"]]

        prepare_inputs_calls = []
        self_calls = []
        update_kwargs_calls = []

        scores = []
        attentions = []
        hidden_states = []

        for i, m_to_fill in enumerate(want_measurements_to_fill):
            if i == 0:
                dep_graph_el_generation_target = None
            else:
                dep_graph_el_generation_target = i

            prepare_inputs_calls.append(
                call(
                    batch,
                    dep_graph_el_generation_target=dep_graph_el_generation_target,
                    **model_kwargs,
                )
            )
            self_calls.append(call(**{f"prepared_{i}": True}, return_dict=True, is_generation=True))
            output = model_outputs[i]
            update_kwargs_calls.append(call(output, model_kwargs))
            model_kwargs = {"updated_{i}": True}

            output.preds.slice.assert_called_once_with((slice(None), -1))
            pred = output.preds.slice.return_value

            scores.append(pred)
            attentions.append(output.attentions)
            hidden_states.append(output.hidden_states)

            pred.sample.assert_called_once_with(batch.event_mask)
            sample = pred.sample.return_value

            if i == 0:
                sample.append_to_batch.assert_called_once_with(batch, M.config)
                sample.update_last_event_data.assert_not_called()
                batch = sample.append_to_batch.return_value
            else:
                sample.append_to_batch.assert_not_called()
                sample.update_last_event_data.assert_called_once_with(
                    batch,
                    M.config,
                    measurements_to_fill=m_to_fill,
                )
                batch = sample.update_last_event_data.return_value

        self.assertNestedCalledWith(M.prepare_inputs_for_generation, prepare_inputs_calls)
        M.assert_has_calls(self_calls, any_order=True)
        self.assertNestedCalledWith(M._update_model_kwargs_for_generation, update_kwargs_calls)

        scores = tuple(scores)
        attentions = tuple(attentions)
        hidden_states = tuple(hidden_states)
        self.assertNestedEqual((batch, scores, attentions, hidden_states, model_kwargs), got)

    def test_nested_attention_sample_event_non_first_event(self):
        class MockMixin(StructuredGenerationMixin, MagicMock):
            def __init__(self, *args, **kwargs):
                StructuredGenerationMixin.__init__(self)
                MagicMock.__init__(self, *args, **kwargs)

        model_outputs = [MagicMock() for _ in range(100)]
        M = MockMixin(side_effect=model_outputs)
        M.config.measurements_per_dep_graph_level = [[], ["foo", "bar"], ["baz"]]

        M.prepare_inputs_for_generation = Mock(side_effect=[{f"prepared_{i}": True} for i in range(10)])
        M._update_model_kwargs_for_generation = Mock(side_effect=[{"updated_{i}": True} for _ in range(10)])

        batch = MagicMock()
        model_kwargs = {"foo": "bar"}

        got = M._nested_attention_sample_event(batch, 3, **model_kwargs)

        want_measurements_to_fill = [{"time"}, ["foo", "bar"], ["baz"]]

        prepare_inputs_calls = []
        self_calls = []
        update_kwargs_calls = []

        scores = []
        attentions = []
        hidden_states = []

        for i, m_to_fill in enumerate(want_measurements_to_fill):
            prepare_inputs_calls.append(call(batch, dep_graph_el_generation_target=i, **model_kwargs))
            self_calls.append(call(**{f"prepared_{i}": True}, return_dict=True, is_generation=True))
            output = model_outputs[i]
            update_kwargs_calls.append(call(output, model_kwargs))
            model_kwargs = {"updated_{i}": True}

            output.preds.slice.assert_called_once_with((slice(None), -1))
            pred = output.preds.slice.return_value

            scores.append(pred)
            attentions.append(output.attentions)
            hidden_states.append(output.hidden_states)

            pred.sample.assert_called_once_with(batch.event_mask)
            sample = pred.sample.return_value

            if i == 0:
                sample.append_to_batch.assert_called_once_with(batch, M.config)
                sample.update_last_event_data.assert_not_called()
                batch = sample.append_to_batch.return_value
            else:
                sample.append_to_batch.assert_not_called()
                sample.update_last_event_data.assert_called_once_with(
                    batch,
                    M.config,
                    measurements_to_fill=m_to_fill,
                )
                batch = sample.update_last_event_data.return_value

        self.assertNestedCalledWith(M.prepare_inputs_for_generation, prepare_inputs_calls)
        M.assert_has_calls(self_calls, any_order=True)
        self.assertNestedCalledWith(M._update_model_kwargs_for_generation, update_kwargs_calls)

        scores = tuple(scores)
        attentions = tuple(attentions)
        hidden_states = tuple(hidden_states)
        self.assertNestedEqual((batch, scores, attentions, hidden_states, model_kwargs), got)


if __name__ == "__main__":
    unittest.main()
