import random
import re

def modify_question(question):
    """
    Modify the question by inserting a random letter into a randomly selected word 
    that contains at least 4 alphabetical characters.
    """
    # Split the question into words while keeping punctuation intact
    words = re.findall(r"\b\w{4,}\b", question)  

    if not words:
        return question  # If no suitable word is found, return the original question

    # Choose a random word from the list of words with at least 4 characters
    chosen_word = random.choice(words)

    # Choose a random position within the chosen word
    insert_position = random.randint(1, len(chosen_word) - 2)  

    # Generate a random lowercase letter
    random_letter = random.choice("abcdefghijklmnopqrstuvwxyz")

    # Insert the random letter into the chosen word
    modified_word = (
        chosen_word[:insert_position] + random_letter + chosen_word[insert_position:]
    )

    # Replace only the first occurrence of the chosen word in the question
    modified_question = question.replace(chosen_word, modified_word, 1)

    return modified_question
