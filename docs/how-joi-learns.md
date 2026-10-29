# How Joi learns

Think of Joi as a program that repeatedly guesses the next character. Initially
its guesses are almost random. Training adjusts numbers inside the program so
that characters from the examples become more likely.

For example, given `The cat sa`, a useful next guess might be `t`. Joi produces
a probability for every character in its vocabulary, rather than one fixed
answer. During generation we sample from those probabilities and append the
chosen character, then repeat.

## The pieces

1. **Training text:** existing short stories. This is a collection of examples,
   often called a corpus or dataset; a database server is not needed.
2. **Tokenizer:** maps characters to numbers. Those numbers identify characters;
   they do not make one letter mathematically greater or more important.
3. **Embeddings:** learned vectors that turn character IDs and positions into
   inputs the layers can process.
4. **Transformer layers:** attention mixes information from previous characters;
   feed-forward networks transform it. Causal masking prevents reading future
   characters while learning to predict them.
5. **Weights and biases:** adjustable numbers used in those calculations. Together
   with embeddings and normalization parameters, they are the model's parameters.
6. **Loss:** a score that penalizes assigning low probability to the actual next
   character in the training example.
7. **Backpropagation:** calculates how each parameter contributed to that loss.
   PyTorch performs this differentiation automatically.
8. **Optimizer:** AdamW uses those gradients to make small parameter updates.

One update uses a **batch** of 16 snippets. Each snippet has 128 input characters
and the corresponding next-character targets. Training for 1,500 updates does
not mean reading the dataset exactly 1,500 times: snippets are sampled, and some
can overlap or repeat.

## Training versus using the model

During **training**, we show the correct next characters, measure errors, and
change parameters. During **generation**, parameters stay fixed while Joi
predicts characters from a prompt. Generation is also called inference.

A **checkpoint** is a file containing learned parameters, architecture settings,
and the vocabulary. Joi's training checkpoints also contain the optimizer and
random-number states, so we can continue where we saved.

We keep **validation** stories separate. They help us check whether prediction
improves on unseen examples. If training loss falls while validation loss rises,
the model may be memorizing patterns too specific to its training data.

## Where the code lives

| File | Responsibility |
| --- | --- |
| `joi/prepare.py` | Download a bounded dataset subset and prepare the splits |
| `joi/tokenizer.py` | Convert characters to IDs and back |
| `joi/model.py` | Define embeddings, causal attention, layers, and generation |
| `joi/train.py` | Sample batches, update parameters, measure loss, save progress |
| `joi/checkpoint.py` | Save and reload model and training state |
| `joi/generate.py` | Continue a prompt using a local checkpoint |
| `joi/evaluate.py` | Score validation examples and save sample continuations |
| `configs/tiny.json` | Choose the model's size and context length |

## What this first experiment can tell us

Falling validation loss shows improved next-character prediction on the sampled
held-out stories. It does not establish reliable reasoning or conversation.
TinyStories is a story dataset, not a direct demonstration of how to respond to
a user. The first outputs may contain recognizable words alongside repetition,
invented words, broken grammar, and unrelated content.

We will inspect what actually happens and improve one variable at a time.
More layers, more data, and longer training each cost more computation, and none
alone guarantees the behavior we want.
