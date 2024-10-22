import nltk
import string
import re

nltk.download('stopwords')
nltk.download('words')
nltk.download('punkt')


class Preprocessor():

    def __init__(self):
        # english stopwords
        self.english_stopwords = nltk.corpus.stopwords.words("english")
        # get list of English words
        self.english_words = nltk.corpus.words.words()
        # create stemmer object
        self.stemmer = nltk.stem.PorterStemmer()


    # Text manipulation part
    def remove_underscore(self, text: str) -> str:
        return text.replace('_', ' ')
    
    def remove_hyphen(self, text: str) -> str:
        return text.replace('-', ' ')

    def remove_extra_whitespace(self, text: str) -> str:
        # remove leading and trailing white space
        text = text.strip()

        # replace multiple consecutive white space characters with a single space
        text = " ".join(text.split())
        return text

    def remove_urls(self, text: str) -> str:
        # regular expression patern to match URLs
        return re.sub(r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?", '', text)
    
    def remove_html_tags(self, text: str) -> str:
        # regular expression pattern to match HTML tags
        return re.sub(r"<[^>]+>", '', text)

    def remove_non_ascii(self, text: str) -> str:
        # Replace any character that is not in the ASCII range with an empty string
        return re.sub(r"[^\x00-\x7F]+", '', text)

    def smart_camelcase_token_split(self, token: str) -> str:

        split_tokens = []
        current_token = ""
        # Iterate over each character in the token
        for i, char in enumerate(token):
            if char.isupper():
                if i > 0 and token[i-1].islower():
                    # If the previous character is lowercase, split here
                    split_tokens.append(current_token)
                    current_token = char
                elif i < len(token) - 1 and token[i+1].islower():
                    # If the next character is lowercase, split here
                    split_tokens.append(current_token)
                    current_token = char
                else:
                    # Otherwise, just add the character to the current token
                    current_token += char
            else:
                # If the character is not uppercase, continue adding to current token
                current_token += char
        
        # Append the last processed token
        if current_token:
            split_tokens.append(current_token)
        
        return split_tokens


    # Token manipulation part
    def tokenize(self, text: str) -> list[str]:
        return(nltk.word_tokenize(text))
        
    def token_isalpha_filter(self, tokens: list[str]) -> list[str]:
        cleaned_tokens = []
        for token in tokens:
            # Split at the apostrophe and take the part before it (if it exists)
            token = token.split("'")[0] if "'" in token else token

            # Keep only alphabetic characters
            cleaned_token = ''.join([char for char in token if char.isalpha()])
            
            # If the cleaned token is not empty, process it using the smart split logic
            if cleaned_token:
                split_result = self.smart_camelcase_token_split(cleaned_token)
                cleaned_tokens.extend(split_result)
        
        # Remove any empty tokens
        cleaned_tokens = [token for token in cleaned_tokens if token]
        return cleaned_tokens

    def lowercase_tokens(self, tokens: list[str]) -> list[str]:
        return [token.lower() for token in tokens]

    def remove_punctuation(self, tokens: list[str]) -> list[str]:
        return [token for token in tokens if token not in string.punctuation]

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        return [token for token in tokens if token not in self.english_stopwords]

    def remove_one_char_tokens(self, tokens: list[str]) -> list[str]:
        return [token for token in tokens if len(token) > 1]
    
    # We don't use this since it is extremely slow
    def correct_english_words(self, tokens: list[str]) -> list[str]:
        corrected_tokens = []
        for token in tokens:
            # find the word with the lowest edit distance
            corrected_token = min(self.english_words, key=lambda x: nltk.edit_distance(x, token))
            corrected_tokens.append(corrected_token)
        return corrected_tokens

    def stem_tokens(self, tokens: list[str]) -> list[str]:
        return [self.stemmer.stem(token) for token in tokens]

    # Actual preprocessing part
    def preprocess(self, text: str) -> list[str]:
        text = self.remove_underscore(text)
        text = self.remove_hyphen(text)
        text = self.remove_extra_whitespace(text)
        text = self.remove_urls(text)
        text = self.remove_html_tags(text)
        text = self.remove_non_ascii(text)
        tokens = self.tokenize(text)
        tokens = self.token_isalpha_filter(tokens)
        tokens = self.lowercase_tokens(tokens)
        tokens = self.remove_punctuation(tokens)
        tokens = self.remove_stopwords(tokens)
        tokens = self.remove_one_char_tokens(tokens)
        # tokens = self.correct_english_words(tokens)
        tokens = self.stem_tokens(tokens)
        return tokens