import pytest
from pydantic import ValidationError

from schemas import ImageScores, ProductAnalysis, UploadResponse


@pytest.mark.parametrize("worn_on", ["ear", "neck", "wrist", "finger", "face", "none"])
@pytest.mark.parametrize("confidence", [0, 0.6, 1])
def test_valid_facts(facts_data, worn_on, confidence):
    facts_data.update(worn_on=worn_on, confidence=confidence,
                      single_product=False, clean_packshot=False)
    facts = ProductAnalysis.model_validate(facts_data, strict=True)
    assert facts.model_dump() == facts_data


@pytest.mark.parametrize("field,value", [
    ("single_product", "true"), ("single_product", 1),
    ("clean_packshot", None), ("sits_in_room", "false"),
    ("product_type", 123), ("worn_on", "head"),
    ("confidence", -0.01), ("confidence", 1.01),
    ("confidence", "0.8"), ("confidence", float("nan")),
    ("confidence", float("inf")), ("unexpected", True),
])
def test_invalid_facts(facts_data, field, value):
    facts_data[field] = value
    with pytest.raises(ValidationError):
        ProductAnalysis.model_validate(facts_data, strict=True)


@pytest.mark.parametrize("field", ProductAnalysis.model_fields)
def test_every_fact_is_required(facts_data, field):
    del facts_data[field]
    with pytest.raises(ValidationError):
        ProductAnalysis.model_validate(facts_data, strict=True)


@pytest.mark.parametrize("status", ["pass", "fail", "needs_review"])
@pytest.mark.parametrize("has_facts", [True, False])
def test_response_round_trip(facts_data, status, has_facts):
    response = UploadResponse(
        status=status, reasons=[],
        scores=ImageScores(min_side=None, sharpness=None),
        facts=ProductAnalysis(**facts_data) if has_facts else None,
    )
    assert UploadResponse.model_validate_json(response.model_dump_json()) == response


def test_response_rejects_unknown_status():
    with pytest.raises(ValidationError):
        UploadResponse(status="success", reasons=[],
                       scores={"min_side": 1000, "sharpness": 50}, facts=None)


@pytest.mark.parametrize("field", ["min_side", "sharpness"])
def test_scores_reject_non_numeric_values(field):
    scores = {"min_side": 1000, "sharpness": 50.0, field: "invalid"}
    with pytest.raises(ValidationError):
        ImageScores(**scores)
