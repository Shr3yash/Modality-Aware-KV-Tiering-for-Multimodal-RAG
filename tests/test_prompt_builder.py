from tests.phase2_spec_helpers import (
    call_with_supported_args,
    get_roles,
    get_value,
    normalize_prompt_result,
    require_any_attr,
    require_phase2_module,
)


def _make_request_and_chunks(image_path: str | None = "/tmp/example.png") -> tuple[object, list[object]]:
    schemas = require_phase2_module("src.rag.schemas")
    generate_request = require_any_attr(schemas, ["GenerateRequest"])
    retrieved_chunk = require_any_attr(schemas, ["RetrievedChunk"])

    request = generate_request(
        query="Explain the cache behavior in this image.",
        image_path=image_path,
        top_k=2,
    )
    chunks = [
        retrieved_chunk(chunk_id="c1", text="cache text context", source="doc1.txt", retrieval_rank=1, modality="text"),
        retrieved_chunk(
            chunk_id="c2",
            text="diagram showing cache tiers",
            source="doc2.txt#page=2",
            retrieval_rank=2,
            modality="image",
            image_path="/tmp/retrieved.png",
        ),
    ]
    return request, chunks


def test_prompt_builder_orders_segments_and_supports_visual_input() -> None:
    prompt_builder_module = require_phase2_module("src.rag.prompt_builder")
    build_prompt = require_any_attr(prompt_builder_module, ["build_prompt", "assemble_prompt"])
    request, chunks = _make_request_and_chunks()

    result = normalize_prompt_result(
        call_with_supported_args(
            build_prompt,
            request=request,
            query=get_value(request, "query"),
            retrieved_chunks=chunks,
            system_instruction="You are a helpful multimodal RAG assistant.",
            image_path=get_value(request, "image_path"),
        )
    )

    segments = result["segments"]
    roles = get_roles(segments)

    assert roles[0] == "system"
    assert "user" in roles
    assert "retrieved_text" in roles
    assert "retrieved_visual" in roles
    assert "visual_context" in roles
    assert roles.index("retrieved_text") > roles.index("user")
    assert result["payload"] is not None
    user_content = result["payload"]["messages"][1]["content"]
    image_parts = [part for part in user_content if part.get("type") == "image_url"]
    assert len(image_parts) == 2


def test_prompt_builder_emits_segment_metadata_for_logging_without_image() -> None:
    prompt_builder_module = require_phase2_module("src.rag.prompt_builder")
    build_prompt = require_any_attr(prompt_builder_module, ["build_prompt", "assemble_prompt"])
    request, chunks = _make_request_and_chunks(image_path=None)

    result = normalize_prompt_result(
        call_with_supported_args(
            build_prompt,
            request=request,
            query=get_value(request, "query"),
            retrieved_chunks=chunks,
            system_instruction="You are a helpful multimodal RAG assistant.",
            image_path=None,
        )
    )

    segments = result["segments"]
    roles = get_roles(segments)

    assert "visual_context" not in roles
    assert "retrieved_visual" in roles
    for segment in segments:
        assert get_value(segment, "role")
        assert get_value(segment, "modality")
        assert get_value(segment, "source")
        assert isinstance(get_value(segment, "text_length"), int)

    retrieved_segments = [segment for segment in segments if get_value(segment, "role") == "retrieved_text"]
    assert retrieved_segments, "Expected at least one retrieved_text segment."
    assert get_value(retrieved_segments[0], "retrieval_rank") == 1
