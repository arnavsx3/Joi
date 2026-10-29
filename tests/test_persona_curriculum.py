import unittest

from joi.prepare_persona import RESERVED_NAMES, make_names, recall_conversations


class PersonaCurriculumTests(unittest.TestCase):
    def test_generated_names_are_disjoint_and_exclude_development_targets(self):
        names=make_names(1400)
        self.assertEqual(names,make_names(1400))
        self.assertEqual(len(names),len(set(names)))
        self.assertFalse({name.lower() for name in names} & RESERVED_NAMES)
        self.assertFalse(set(names[:1200]) & set(names[1200:]))

    def test_training_target_copies_the_supplied_name_with_varied_questions(self):
        names=make_names(20)
        examples=recall_conversations(names,21)
        self.assertEqual(examples,recall_conversations(names,21))
        for name,conversation in zip(names,examples):
            lines=conversation.strip().splitlines()
            self.assertTrue(lines[-1].startswith("Joi:"))
            self.assertIn(name,lines[1])
            self.assertIn(name,lines[-1])


if __name__ == "__main__":
    unittest.main()
