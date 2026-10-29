import pytest
import torch
import torch.nn.functional as F
from pydantic import ValidationError

from item_compare.calibration import Calibration, Logistic
from item_compare.schemas import Item, Kind
from item_compare.similarity import ItemEmbedding, fuse, raw_scores

WEIGHTS = {Kind.IMAGE_IMAGE: 0.6, Kind.TEXT_TEXT: 0.4, Kind.IMAGE_TEXT: 1.0}


def vec(*values: float) -> torch.Tensor:
    return F.normalize(torch.tensor(values, dtype=torch.float32), dim=-1)


def test_item_needs_image_or_text():
    with pytest.raises(ValidationError):
        Item(id="empty")


def test_item_text_puts_title_and_attributes_before_description():
    item = Item(title="Air Max 90", description="Classic sneaker", attributes={"brand": "Nike", "size": 42})
    assert item.text() == "Air Max 90. brand: Nike. size: 42. Classic sneaker"


def test_item_ignores_unknown_fields():
    item = Item.model_validate({"title": "Mug", "price": 9.99, "sku": "M-1"})
    assert item.text() == "Mug"


def test_logistic_is_50_percent_at_shift_and_monotonic():
    curve = Logistic(scale=30, shift=0.85)
    assert curve(0.85) == pytest.approx(50)
    assert curve(0.6) < curve(0.85) < curve(0.95) < 100


def test_raw_scores_compares_shared_parts():
    a = ItemEmbedding(image=vec(1, 0), text=vec(0, 1))
    b = ItemEmbedding(image=vec(1, 0), text=vec(1, 0))
    assert raw_scores(a, b) == [(Kind.IMAGE_IMAGE, pytest.approx(1.0)), (Kind.TEXT_TEXT, pytest.approx(0.0))]


@pytest.mark.parametrize("swap", [False, True])
def test_raw_scores_falls_back_to_image_vs_text(swap):
    a = ItemEmbedding(image=vec(1, 1))
    b = ItemEmbedding(text=vec(1, 0))
    a, b = (b, a) if swap else (a, b)
    [(kind, cosine)] = raw_scores(a, b)
    assert kind is Kind.IMAGE_TEXT
    assert cosine == pytest.approx(2**-0.5)


def test_fuse_renormalizes_weights_of_present_parts():
    result = fuse([(Kind.TEXT_TEXT, 0.99)], Calibration(), WEIGHTS, threshold=50)
    assert result.components[0].weight == 1.0
    assert result.similarity == result.components[0].similarity
    assert result.is_same


def test_fuse_is_weighted_mean():
    calibration = Calibration({Kind.IMAGE_IMAGE: Logistic(1e3, 0.5), Kind.TEXT_TEXT: Logistic(1e3, 0.5)})
    result = fuse([(Kind.IMAGE_IMAGE, 0.9), (Kind.TEXT_TEXT, 0.1)], calibration, WEIGHTS, threshold=50)
    assert result.similarity == pytest.approx(60.0)  # 0.6 * 100 + 0.4 * 0


def test_calibration_roundtrip(tmp_path):
    path = tmp_path / "calibration.json"
    Calibration({Kind.TEXT_TEXT: Logistic(12.5, 0.7)}).save(path)
    loaded = Calibration.load(path)
    assert loaded.curves[Kind.TEXT_TEXT] == Logistic(12.5, 0.7)
    assert loaded.curves[Kind.IMAGE_IMAGE] == Calibration().curves[Kind.IMAGE_IMAGE]
