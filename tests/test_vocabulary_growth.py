import unittest

import torch

from joi.expand_vocabulary import expand_model
from joi.model import Joi, ModelConfig
from joi.tokenizer import WordTokenizer


class VocabularyGrowthTests(unittest.TestCase):
    def test_preserves_learned_parameters_and_initializes_new_words_from_own_pieces(self):
        torch.set_num_threads(1)
        old = WordTokenizer.from_text("hello hello", 100)
        new = WordTokenizer(old.chars + ["painting"])
        source = Joi(ModelConfig(len(old.chars), block_size=16, n_embd=8, n_heads=2, n_layers=1, dropout=0))
        grown = expand_model(source, old, new)
        for key, value in source.state_dict().items():
            if key in ("token_embedding.weight", "head.weight"):
                self.assertTrue(torch.equal(value, grown.state_dict()[key][:len(old.chars)]))
            else:
                self.assertTrue(torch.equal(value, grown.state_dict()[key]))
        expected = source.token_embedding.weight[old.encode("painting")].mean(0)
        torch.testing.assert_close(grown.token_embedding.weight[-1], expected)
        self.assertIs(grown.head.weight, grown.token_embedding.weight)
        tokens = torch.tensor([old.encode("hello")])
        before, _ = source(tokens)
        after, _ = grown(tokens)
        torch.testing.assert_close(before, after[:, :, :len(old.chars)])
        self.assertEqual(new.decode(new.encode("painting")), "painting")


if __name__ == "__main__":
    unittest.main()
