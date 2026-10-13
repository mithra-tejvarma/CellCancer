import pytest

from src.bccd import parse_voc
from src.preprocessing import normalize_box, denormalize_box, wbc_labels


def test_xml_parsing_uses_image_size_and_ignores_embedded_path(synthetic_bccd):
    root, _, _ = synthetic_bccd
    record = parse_voc(root / "Annotations/BloodImage_00000.xml", root / "JPEGImages/BloodImage_00000.jpg")
    assert (record["width"],record["height"]) == (100,80)
    assert record["xml_size"] == ["999","999"]
    assert {o["class"] for o in record["objects"]} == {"RBC","WBC","Platelets"}
    assert wbc_labels(record) == [[0, .4, .375, .4, .5]]


def test_wbc_filter_and_empty_label(synthetic_bccd):
    root, _, _ = synthetic_bccd
    record = parse_voc(root / "Annotations/BloodImage_00003.xml", root / "JPEGImages/BloodImage_00003.jpg")
    assert wbc_labels(record) == []


def test_normalization_roundtrip_and_invalid_boxes():
    values = normalize_box([10,20,90,70],100,80)
    assert denormalize_box(values,100,80) == pytest.approx([10,20,90,70])
    for box in ([5,5,5,8], [-1,1,5,5], [0,0,101,5], [float("nan"),1,5,6]):
        with pytest.raises(ValueError):
            normalize_box(box,100,80)


def test_missing_annotation_and_invalid_xml_box(synthetic_bccd):
    root, _, _ = synthetic_bccd
    image = root / "JPEGImages/BloodImage_00000.jpg"
    with pytest.raises(FileNotFoundError):
        parse_voc(root / "Annotations/missing.xml", image)
    xml = root / "Annotations/BloodImage_00000.xml"
    xml.write_text(xml.read_text().replace("<xmax>60</xmax>","<xmax>20</xmax>"),encoding="utf-8")
    with pytest.raises(ValueError, match="nonpositive"):
        parse_voc(xml,image)
