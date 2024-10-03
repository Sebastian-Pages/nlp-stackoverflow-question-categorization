import sys
import os

# Dynamically add the src directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


from nlp.cleaning import (
    lower_case,
    remove_spaces_tabs,
    remove_punct,
    remove_non_ascii,
    remove_single_char,
    remove_tags,
    remove_url,
    remove_digits,
    expand_contractions,
)


# Test for lower_case function
def test_lower_case():
    assert lower_case("Hello World!") == "hello world!"
    assert lower_case("PYTEST") == "pytest"
    assert lower_case("123 ABC") == "123 abc"


# Test for remove_spaces_tabs function
def test_remove_spaces_tabs():
    assert remove_spaces_tabs("Hello   World!") == "Hello World!"
    assert remove_spaces_tabs("   Leading and trailing   ") == "Leading and trailing"
    assert remove_spaces_tabs("\tTab \t and \n new line") == "Tab and new line"


# Test for remove_punct function
def test_remove_punct():
    assert remove_punct("Hello, World!") == "Hello World"
    assert remove_punct("No punctuation!") == "No punctuation"
    assert (
        remove_punct("Text with. lots; of punctuation?!")
        == "Text with lots of punctuation"
    )


# Test for remove_non_ascii function
def test_remove_non_ascii():
    assert remove_non_ascii("Café") == "Caf"
    assert remove_non_ascii("naïve") == "nave"
    assert remove_non_ascii("ASCII only!") == "ASCII only!"


# Test for remove_single_char function
def test_remove_single_char():
    assert remove_single_char("This is a test sentence.") == "This is test sentence."
    assert remove_single_char("A B C D E F G") == ""
    assert remove_single_char("Keep C", keep="C") == "Keep C"


# Test for remove_tags function
def test_remove_tags():
    assert remove_tags("<html><body>Hello World!</body></html>") == "Hello World!"
    assert remove_tags("<div>Text with <b>bold</b> tag</div>") == "Text with bold tag"
    assert remove_tags("<p>This is a <a href='#'>link</a>.</p>") == "This is a link."


# Test for remove_url function
def test_remove_url():
    assert remove_url("Visit https://example.com") == "Visit "
    assert remove_url("Check www.example.com ok") == "Check  ok"
    assert remove_url("No URL here.") == "No URL here."


# Test for remove_digits function
def test_remove_digits():
    assert remove_digits("123abc456") == "abc"
    assert remove_digits("No digits here") == "No digits here"
    assert remove_digits("2023 is the year!") == " is the year!"


# Test for expand_contractions function
def test_expand_contractions():
    assert expand_contractions("I'm here.") == "I am here."
    assert expand_contractions("Don't worry.") == "Do not worry."
    assert expand_contractions("He's a good person.") == "He is a good person."
