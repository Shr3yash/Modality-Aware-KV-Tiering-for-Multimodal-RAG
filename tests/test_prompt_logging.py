from tests.phase2_spec_helpers import call_with_supported_args, get_value, require_any_attr, require_phase2_module


def test_prompt_logging_summarizes_roles_modalities_and_token_counts() -> None:
    prompt_logging_module = require_phase2_module("src.utils.prompt_logging")
    summarize_prompt = require_any_attr(
        prompt_logging_module,
        ["summarize_prompt", "summarize_prompt_segments"],
    )
    segments = [
        {
            "role": "system",
            "modality": "text",
            "source": "system",
            "text": "You are helpful.",
            "text_length": 16,
        },
        {
            "role": "user",
            "modality": "text",
            "source": "user",
            "text": "Explain the cache.",
            "text_length": 18,
        },
        {
            "role": "retrieved_text",
            "modality": "text",
            "source": "doc1.txt",
            "text": "cache context from corpus",
            "text_length": 25,
            "retrieval_rank": 1,
        },
        {
            "role": "retrieved_visual",
            "modality": "image",
            "source": "doc1.txt#page=1",
            "text": "chart about cache tiers",
            "text_length": 23,
            "retrieval_rank": 2,
        },
        {
            "role": "visual_context",
            "modality": "image",
            "source": "/tmp/example.png",
            "text": "",
            "text_length": 0,
        },
    ]

    summary = call_with_supported_args(
        summarize_prompt,
        segments=segments,
        prompt_segments=segments,
        token_counter=lambda text: len(text.split()),
    )

    assert get_value(summary, "total_prompt_tokens") >= 7
    assert get_value(summary, "retrieved_text_token_count", "retrieved_tokens") >= 4
    assert get_value(summary, "visual_input_present", "has_visual_input") is True
    assert get_value(summary, "num_retrieved_chunks", "retrieved_chunk_count") == 2
    assert get_value(summary, "ordered_segment_roles", "segment_roles") == [
        "system",
        "user",
        "retrieved_text",
        "retrieved_visual",
        "visual_context",
    ]
