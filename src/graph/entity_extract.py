"""
src/graph/entity_extract.py
Extract the bracketed topic entity from MetaQA questions.
Example: "What movies did [Tom Hanks] star in?" → "Tom Hanks"
"""

import re


def extract_topic_entity(question: str) -> str | None:
    """
    Extract the topic entity from a MetaQA question.
    MetaQA always brackets exactly one entity per question.

    Returns:
        The entity string, or None if no bracket found.
    """
    match = re.search(r'\[(.+?)\]', question)
    return match.group(1) if match else None


def clean_question(question: str) -> str:
    """
    Remove the brackets from the question for display/prompting.
    Example: "What movies did [Tom Hanks] star in?"
           → "What movies did Tom Hanks star in?"
    """
    return re.sub(r'\[(.+?)\]', r'\1', question)


if __name__ == "__main__":
    examples = [
        "What movies did [Tom Hanks] star in?",
        "Who directed [The Matrix]?",
        "What is the genre of movies directed by the director of [Inception]?",
    ]
    for q in examples:
        entity = extract_topic_entity(q)
        clean = clean_question(q)
        print(f"Q      : {q}")
        print(f"Entity : {entity}")
        print(f"Clean  : {clean}")
        print()
