import unittest
import torch
import torch.nn as nn
from src.model_query import (
    TargetAwareCrossAttention,
    ComponentInteractionLayer,
    TargetAwareQueryAttentionModel,
)
from src.losses import ccc_loss, GaussLoss


class TestTargetAwareQueryAttention(unittest.TestCase):

    def test_target_aware_cross_attention_shapes(self):
        B, S, H = 2, 16, 768
        h_final = torch.randn(B, S, H)
        attention_mask = torch.ones(B, S, dtype=torch.long)
        attention_mask[:, 12:] = 0  # last 4 tokens are padding

        layer1 = TargetAwareCrossAttention(hidden_size=H, num_heads=4, dropout=0.1)
        z, attn_weights = layer1(h_final, attention_mask=attention_mask)

        self.assertEqual(z.shape, (B, 3, H))
        self.assertEqual(attn_weights.shape, (B, 3, S))
        # Check that attention to padded tokens is zero or near zero
        self.assertTrue(torch.all(attn_weights[:, :, 12:] <= 1e-4))

    def test_component_interaction_shapes(self):
        B, H = 2, 768
        z = torch.randn(B, 3, H)
        
        layer2 = ComponentInteractionLayer(hidden_size=H, num_heads=4, dropout=0.1)
        z_prime, self_attn_weights = layer2(z)

        self.assertEqual(z_prime.shape, (B, 3, H))
        self.assertEqual(self_attn_weights.shape, (B, 3, 3))

    def test_query_attention_role_ids_and_heads(self):
        B, S, H = 2, 10, 768
        input_ids = torch.randint(100, 2000, (B, S))
        attention_mask = torch.ones(B, S, dtype=torch.long)
        mod_span_mask = torch.zeros(B, S, dtype=torch.bool)
        head_span_mask = torch.zeros(B, S, dtype=torch.bool)
        
        # Set span masks for item 0
        mod_span_mask[0, 2:4] = True
        head_span_mask[0, 4:6] = True

        # Set span masks for item 1
        mod_span_mask[1, 1:2] = True
        head_span_mask[1, 2:3] = True

        # Create model components directly
        role_embed = nn.Embedding(4, H)
        layer1 = TargetAwareCrossAttention(hidden_size=H, num_heads=4, dropout=0.1)
        layer2 = ComponentInteractionLayer(hidden_size=H, num_heads=4, dropout=0.1)
        
        from src.heads import GaussHead
        shared_gauss = GaussHead(in_features=H, hidden=64, floor=0.04)

        # Role IDs mapping
        role_ids = torch.zeros_like(input_ids, dtype=torch.long)
        role_ids = torch.where(mod_span_mask, torch.full_like(role_ids, 1), role_ids)
        role_ids = torch.where(head_span_mask, torch.full_like(role_ids, 2), role_ids)

        self.assertEqual(role_ids[0, 2].item(), 1)
        self.assertEqual(role_ids[0, 4].item(), 2)
        self.assertEqual(role_ids[0, 0].item(), 0)

        # Forward pass without full transformer download
        h_dummy = torch.randn(B, S, H)
        e_role = role_embed(role_ids)
        h_final = h_dummy + e_role
        
        z, cross_attn_map = layer1(h_final, attention_mask)
        z_prime, comp_attn_map = layer2(z)
        
        mod_mu, mod_sigma = shared_gauss(z_prime[:, 0, :])
        head_mu, head_sigma = shared_gauss(z_prime[:, 1, :])
        comp_mu, comp_sigma = shared_gauss(z_prime[:, 2, :])

        self.assertEqual(mod_mu.shape, (B,))
        self.assertEqual(mod_sigma.shape, (B,))
        self.assertTrue(torch.all(mod_sigma >= 0.04))

    def test_ccc_loss_with_normalized_targets(self):
        # Targets normalized: y_norm in [0.2, 1.0]
        y_raw = torch.tensor([4.5, 2.0, 1.2, 3.8, 5.0])
        y_norm = y_raw / 5.0

        # Predictions
        pred_norm = torch.tensor([4.4, 2.1, 1.0, 3.9, 4.8]) / 5.0
        loss = ccc_loss(pred_norm, y_norm)

        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss).item())


if __name__ == '__main__':
    unittest.main()
