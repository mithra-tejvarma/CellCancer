import io
import unittest

from PIL import Image

from src.integrated_pipeline import clip_box, decode_image, quantify, unique_boxes


class PipelineContractTests(unittest.TestCase):
    def test_full_unfreezes_entire_backbone(self):
        import torch
        from src.classifier_model import build_model, set_trainable
        model = build_model(pretrained=False)
        set_trainable(model, "full")
        self.assertTrue(all(param.requires_grad for param in model.parameters()))
        optimizer = torch.optim.AdamW(param for param in model.parameters() if param.requires_grad)
        covered = {id(param) for group in optimizer.param_groups for param in group["params"]}
        self.assertEqual(covered, {id(param) for param in model.parameters()})

    def test_threshold_uses_validation_labels(self):
        from scripts.full_finetune import select_threshold
        metrics = select_threshold([0, 0, 1, 1], [.1, .3, .7, .9])
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["tn"], 2)
        self.assertEqual(metrics["tp"], 2)

    def test_zero_detection(self):
        self.assertEqual(quantify([])["estimated_blast_fraction_percent"], None)
        self.assertEqual(quantify([])["total_detected_wbc"], 0)

    def test_uncertain_not_counted_as_normal(self):
        cells = [{"label": "Unclassified"}, {"label": "Leukemic Blast (ALL)"},
                 {"label": "Normal (HEM)"}]
        result = quantify(cells, total_detected=4)
        self.assertEqual(result["unclassified"], 2)
        self.assertEqual(result["total_classified_wbc"], 2)
        self.assertEqual(result["estimated_blast_fraction_percent"], 50)

    def test_clipping_and_duplicate_filter(self):
        self.assertEqual(clip_box((-4, 2, 12, 25), 10, 20), (0, 2, 10, 20))
        self.assertIsNone(clip_box((20, 2, 21, 5), 10, 20))
        boxes = [{"box": (0, 0, 10, 10), "detection_confidence": .9},
                 {"box": (1, 1, 10, 10), "detection_confidence": .8}]
        self.assertEqual(len(unique_boxes(boxes)), 1)

    def test_image_decode_and_rejection(self):
        image = Image.new("RGBA", (10, 8), (255, 0, 0, 255))
        stream = io.BytesIO(); image.save(stream, format="PNG")
        decoded = decode_image(stream.getvalue())
        self.assertEqual((decoded.mode, decoded.size), ("RGB", (10, 8)))
        with self.assertRaises(ValueError):
            decode_image(b"damaged")


if __name__ == "__main__":
    unittest.main()
