import unittest

from joi.prepare_persona import RESERVED_NAMES, make_names, recall_conversations
from joi.recall_evaluate import recall_trial


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

    def test_recall_probe_never_supplies_the_reference_assistant_turn(self):
        class Engine:
            seen=[]
            def reply(self,history,**kwargs):
                self.seen.append([dict(message) for message in history])
                return {"reply":"Actual generated greeting." if len(history)==1 else "Your name is Bavodi."}
        engine=Engine()
        conversation=recall_conversations(["Bavodi"],21)[0]
        trial=recall_trial(engine,conversation,1000)
        self.assertTrue(trial["correct"])
        self.assertEqual(engine.seen[1][1]["content"],"Actual generated greeting.")


if __name__ == "__main__":
    unittest.main()
