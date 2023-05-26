import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Load the SentimentIntensityAnalyzer from NLTK
sia = SentimentIntensityAnalyzer()

# Here we define some keywords that might be used in swatting calls
swatting_keywords = ['hostage', 'bomb', 'shoot', 'kill', 'gun']

def analyze_call(call_text):
    # Apply sentiment analysis to the call text
    sentiment = sia.polarity_scores(call_text)

    # Check if any swatting keywords are in the call text
    swatting_terms_present = any(term in call_text for term in swatting_keywords)

    # Here's a very simple heuristic to classify a call as potential swatting:
    # If the call's sentiment is overwhelmingly negative, and any swatting keyword is present,
    # we classify it as potential swatting.
    if sentiment['compound'] < -0.5 and swatting_terms_present:
        return True

    return False

# Test the function with a call
call_text = "There's a man with a gun, he's threatening to kill everyone."
is_swatting = analyze_call(call_text)

print(f"Potential swatting call: {is_swatting}")
