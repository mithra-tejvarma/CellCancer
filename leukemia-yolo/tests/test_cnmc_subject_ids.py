import pytest
from src.cnmc_data import subject_id


def test_official_filename_identity():
    assert subject_id("UID_57_29_1_all.bmp", "all") == "ALL:57"
    assert subject_id("UID_H49_35_1_hem.bmp", "hem") == "HEM:H49"
    assert subject_id("UID_57_29_2_all.bmp", "all") == "ALL:57"


@pytest.mark.parametrize("name,label", [("UID_57_29_1_all.bmp", "hem"),
                                        ("UID_H49_35_1_hem.bmp", "all"),
                                        ("BloodImage_0001.bmp", "all"),
                                        ("UID_0_29_1_all.bmp", "all")])
def test_bad_subject_names_stop(name, label):
    with pytest.raises(ValueError):
        subject_id(name, label)
