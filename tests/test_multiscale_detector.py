import unittest

import torch


class TestMultiscaleDetector(unittest.TestCase):
    def test_image_encoder_extract_stage_features(self):
        from clipdetr.models.image_encoder import ImageEncoder

        encoder = ImageEncoder(
            backbone_name="mobilenet_v3_small",
            pretrained=False,
            output_dim=64,
        )
        x = torch.randn(1, 3, 64, 64)
        stage_features = encoder.extract_stage_features(x, levels=3)

        self.assertEqual(list(stage_features.keys()), ["res3", "res4", "res5"])
        heights = [int(feature.shape[-2]) for feature in stage_features.values()]
        self.assertGreater(heights[0], heights[1])
        self.assertGreater(heights[1], heights[2])

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

    def test_feature_neck_outputs_match_hidden_dim_and_resolution(self):
        from clipdetr.models.feature_neck import LightweightFPNNeck
        from clipdetr.models.image_encoder import ImageEncoder

        encoder = ImageEncoder(
            backbone_name="mobilenet_v3_small",
            pretrained=False,
            output_dim=64,
        )
        x = torch.randn(1, 3, 64, 64)
        stage_features = encoder.extract_stage_features(x, levels=3)
        neck = LightweightFPNNeck(in_channels=[24, 48, 576], out_channels=64)
        outputs = neck(list(stage_features.values()))

        self.assertEqual(len(outputs), 3)
        for output, stage in zip(outputs, stage_features.values()):
            self.assertEqual(int(output.shape[1]), 64)
            self.assertEqual(tuple(output.shape[-2:]), tuple(stage.shape[-2:]))

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

    def test_light_detr_multiscale_neck_forward_shapes(self):
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
            use_multiscale_neck=True,
            multiscale_levels=3,
        )
        x = torch.randn(2, 3, 64, 64)
        outputs = model(x)

        self.assertEqual(tuple(outputs["pred_logits"].shape), (2, 10, 6))
        self.assertEqual(tuple(outputs["pred_boxes"].shape), (2, 10, 4))

    def test_light_detr_multiscale_neck_uses_bounded_memory_tokens(self):
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
            use_multiscale_neck=True,
            multiscale_levels=3,
        )
        x = torch.randn(1, 3, 128, 128)
        raw_stage_features = model.image_encoder.extract_stage_features(x, levels=3)
        raw_token_count = sum(int(feature.shape[-2] * feature.shape[-1]) for feature in raw_stage_features.values())

        memory = model.encode_images(x)
        self.assertLess(int(memory.shape[1]), raw_token_count)
        self.assertEqual(int(memory.shape[2]), 64)


if __name__ == "__main__":
    unittest.main()
