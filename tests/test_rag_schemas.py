from tests.phase2_spec_helpers import declared_fields, require_any_attr, require_phase2_module


def test_phase2_schema_models_expose_minimal_request_and_response_fields() -> None:
    schemas = require_phase2_module("src.rag.schemas")

    generate_request = require_any_attr(schemas, ["GenerateRequest"])
    retrieved_chunk = require_any_attr(schemas, ["RetrievedChunk"])
    prompt_segment = require_any_attr(schemas, ["PromptSegment"])
    generate_response = require_any_attr(schemas, ["GenerateResponse"])

    assert {"query", "image_path", "top_k"} <= declared_fields(generate_request)
    assert {"chunk_id", "text", "source", "modality", "image_path"} <= declared_fields(retrieved_chunk)
    assert {"role", "modality", "source", "text_length"} <= declared_fields(prompt_segment)
    assert {"answer_text", "prompt_summary", "token_summary", "timing_summary"} <= declared_fields(generate_response)
