## Language Model

This is an implementation of a language model using the Transformer architecture. The code is inspired by Assignment 1 of the CS 336 course at Stanford University.

## Personal Notes

- I have implemented contrastive loss for the language model. This does not work properly yet, but I am hoping to spend some time to see if I can get it to work.
    - I have implemented the loss function and I am sure it is correct.
    - With the contrastive objective, I make the observation that the model starts to "context switch" to the middle of another story when trained on TinyStories. My best guess is that since each batch element can span through multiple stories, the model starts to pick up patterns from the other stories as the contrastive loss is meant to encode the future properties of the data.
    - TODO: I want to try this on a bigger dataset and make each batch element only contain a single passage.
- I have also implemented KV-cache for the language model generation.