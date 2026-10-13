from PIL import Image
from src.classifier_gradcam import cam_for_image
from src.classifier_model import build_model


def test_cam_spatial_size_cpu():
    model = build_model(pretrained=False)
    original, heat, overlay = cam_for_image(model, Image.new("RGB", (72, 72), "white"), 224, "cpu", 1)
    assert original.shape == (224, 224, 3)
    assert heat.shape == (224, 224)
    assert overlay.shape == (224, 224, 3)
