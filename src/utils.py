import string
import re


accented_chars = {
    'a': u'a á à ả ã ạ â ấ ầ ẩ ẫ ậ ă ắ ằ ẳ ẵ ặ',
    'o': u'o ó ò ỏ õ ọ ô ố ồ ổ ỗ ộ ơ ớ ờ ở ỡ ợ',
    'e': u'e é è ẻ ẽ ẹ ê ế ề ể ễ ệ',
    'u': u'u ú ù ủ ũ ụ ư ứ ừ ử ữ ự',
    'i': u'i í ì ỉ ĩ ị',
    'y': u'y ý ỳ ỷ ỹ ỵ',
    'd': u'd đ',
}

plain_char_map = {}
for c, variants in accented_chars.items():
    for v in variants.split(' '):
        plain_char_map[v] = c

_alphabets = list(string.digits + string.ascii_lowercase)
for c, variants in accented_chars.items():
    _alphabets.extend(variants.split()[1:])


def remove_punctuation(text):
    """https://stackoverflow.com/a/37221663"""
    table = str.maketrans({key: None for key in string.punctuation})
    return text.translate(table)


def remove_multiple_space(text):
    return re.sub("\s\s+", " ", text)  # noqa


def make_words(text):
    words = re.findall(r'\w+', text.lower())
    words = [remove_punctuation(word) for word in words]
    return words


def extract_phrases(text):
    """ extract phrases, i.e. group of continuous words, from text """
    return re.findall('\w[\w ]+', text, re.UNICODE)  # noqa


def remove_accent(text):
    return u''.join(plain_char_map.get(char, char) for char in text)


def remove_non_alphanum(text, format_=_alphabets):
    return re.sub("[^(?:(?! 0123456789abcdefghijklmnopqrstuvwxyzáàảãạâấầẩẫậăắằẳẵặóòỏõọôốồổỗộơớờởỡợéèẻẽẹêếềểễệúùủũụưứừửữựíìỉĩịýỳỷỹỵđ).)*]", "", text)  # noqa


def preprocessing_text(text):
    texts = extract_phrases(text)
    texts = [t.lower() for t in texts]
    texts = [remove_multiple_space(t) for t in texts]
    texts = [remove_non_alphanum(t) for t in texts]
    return texts


if __name__ == '__main__':
    print(len(_alphabets))
