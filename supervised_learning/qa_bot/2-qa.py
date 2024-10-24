#!/usr/bin/env python3
"""
QA with BERT and user input loop.
"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad")
model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question.
    - Uses the `bert-uncased-tf2-qa` model from the `tensorflow-hub` library
    - Uses the pre-trained `BertTokenizer` from the `transformers` library,
    `bert-large-uncased-whole-word-masking-finetuned-squad`.

    Args:
        question: A string containing the question to answer
        reference: A string containing the reference document from which to
            find the answer

    Returns:
        A string containing the answer. If no answer is found, return `None`
    """
    # Tokenize the input using the BERT tokenizer
    inputs = tokenizer(question, reference, return_tensors="tf")

    # Prepare the input for the TensorFlow Hub model
    input_tensors = [
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["token_type_ids"]
    ]

    # Pass the input tensors to the model
    output = model(input_tensors)

    # Access the start and end logits
    start_logits = output[0]
    end_logits = output[1]

    # Get the input sequence length
    sequence_length = inputs["input_ids"].shape[1]

    # Find the best start and end indices within the input sequence
    start_index = tf.math.argmax(start_logits[0, 1:sequence_length - 1]) + 1
    end_index = tf.math.argmax(end_logits[0, 1:sequence_length - 1]) + 1

    # Get the answer tokens using the best indices
    answer_tokens = inputs["input_ids"][0][start_index: end_index + 1]

    # Decode the answer tokens
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True,
                              clean_up_tokenization_spaces=True)

    # No answer ? Returning None
    if not answer.strip():
        return None

    return answer


def answer_loop(reference):
    """
    Continuously prompts the user for input until one of the predefined
    farewell words is entered. Answers questions from a given reference
    text using a BERT tokenizer and model.
    """

    farewells = ["exit", "quit", "goodbye", "bye"]

    user_input = ""
    while True:
        user_input = input("Q: ")

        # Check user input for farewell string (case insensitive)
        if user_input.lower() in farewells:
            print("A: Goodbye")
            return

        # Response to any other input
        answer = question_answer(question=user_input, reference=reference)
        if answer is None:
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A: {answer}")