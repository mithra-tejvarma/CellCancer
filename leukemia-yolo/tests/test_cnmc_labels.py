from src.cnmc_data import inspect_cnmc


def test_fixed_class_mapping(synthetic_cnmc):
    _, _, _, cfg = synthetic_cnmc
    frame, _ = inspect_cnmc(cfg)
    assert set(frame[frame.class_name == "HEM"].class_id) == {0}
    assert set(frame[frame.class_name == "ALL"].class_id) == {1}
