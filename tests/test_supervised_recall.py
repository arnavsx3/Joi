import unittest

from joi.tokenizer import WordTokenizer
from joi.tune_recall import supervised_example, batch
import random


class SupervisedRecallTests(unittest.TestCase):
    def test_only_reply_tokens_are_targets_and_name_tokens_have_extra_weight(self):
        tokenizer=WordTokenizer.from_text("User My name is Your What would you like to talk about Joi "*3,120)
        prefix="User: My name is Bavodi.\nJoi: "
        target="Your name is Bavodi. What would you like to talk about?"
        tokens,labels,weights=supervised_example(tokenizer,prefix,target,"Bavodi")
        self.assertEqual(len(tokens),len(labels))
        self.assertEqual(len(tokens),len(weights))
        first=len(tokenizer.encode(prefix))-1
        self.assertTrue(all(label==-100 for label in labels[:first]))
        self.assertTrue(all(weight==0 for weight in weights[:first]))
        named=[label for label,weight in zip(labels,weights) if weight==8]
        self.assertEqual(tokenizer.decode(named),"Bavodi")
        x,y,w=batch([(tokens,labels,weights)],2,random.Random(1),256)
        self.assertEqual(x.shape,y.shape)
        self.assertEqual(y.shape,w.shape)


if __name__ == "__main__":
    unittest.main()
