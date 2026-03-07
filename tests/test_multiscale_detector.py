import unittest

import torch


class TestMultiscaleDetector(unittest.TestCase):
    def test_image_encoder_multiscale_produces_more_tokens(self):
        from clipdetr.models.image_encoder import ImageEncoder

        encoder = ImageEncoder(
            backbone_name="mobilenet_v3_small",
            pretrained=False,
            output_dim=64,
            multiscale_levels=3,
        )
        x = torch.randn(1, 3, 64, 64)
        single_scale = encoder(x, return_patch_tokens=True, use_multiscale_tokens=False)
        multi_scale = encoder(x, return_patch_tokens=True, use_multiscale_tokens=True)

        self.assertEqual(int(single_scale.shape[2]), 64)
        self.assertEqual(int(multi_scale.shape[2]), 64)
        self.assertGreater(int(multi_scale.shape[1]), int(single_scale.shape[1]))

    def test_light_detr_multiscale_forward_shapes(self):
        from clipdetr.models.light_detr import LightDETR

        model = LightDETR(
            num_classes=5,
            hidden_dim=64,
            num_queries=10,
            decoder_layers=1,
            num_heads=4,
            ff_dim=128,
            dropout=0.0,
            image_backbone="mobilenet_v3_small",
            image_pretrained=False,
            use_multiscale_memory=True,
            multiscale_levels=3,
        )
        x = torch.randn(2, 3, 64, 64)
        outputs = model(x)

        self.assertEqual(tuple(outputs["pred_logits"].shape), (2, 10, 6))
        self.assertEqual(tuple(outputs["pred_boxes"].shape), (2, 10, 4))


if __name__ == "__main__":
    unittest.main()
