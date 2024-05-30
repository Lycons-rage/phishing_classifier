import re
from urllib.parse import urlparse
from collections import Counter
import math

# need to practice over a sample url for all the changes that are needed to be done

url = "https://account.ineuron.ai/signin?domain=learn.ineuron.ai&redirectUrl=/"

# we gotta maintain the same order of features as in our dataset 
# this is one hefty work 

def entropy(s):
    """Calculate the entropy of a string."""
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * math.log2(count/lns) for count in p.values())


def extract_features(url):
    # Initialize the data_dict
    data_dict = {
        'url_length' : len(url),
        'number_of_dots_in_url' : url.count('.'),
        'having_repeated_digits_in_url': int(bool(re.search(r'(\d)\1', url))),
        'number_of_digits_in_url': len(re.findall(r'\d', url)), 
        'number_of_special_char_in_url': len(re.findall(r'[~`!@#$%^&*()_\-+=\'";:<>,./?|\{}[\]]', url)),
        'number_of_hyphens_in_url': url.count('-'), 
        'number_of_underline_in_url': url.count('_'),
        'number_of_slash_in_url': url.count('/'), 
        'number_of_questionmark_in_url': url.count('?'),
        'number_of_equal_in_url': url.count('='), 
        'number_of_at_in_url': url.count('@'),
        'number_of_dollar_in_url': url.count('$'), 
        'number_of_exclamation_in_url': url.count('!'),
        'number_of_hashtag_in_url': url.count('#'), 
        'number_of_percent_in_url': url.count('%'), 
        'domain_length': len(urlparse(url).netloc),
        'number_of_dots_in_domain': urlparse(url).netloc.count('.'), 
        'number_of_hyphens_in_domain': urlparse(url).netloc.count('-'),
        'having_special_characters_in_domain': int(bool(re.search(r'[^a-zA-Z0-9.-]', urlparse(url).netloc))),
        'number_of_special_characters_in_domain': len(re.findall(r'[^a-zA-Z0-9.-]', urlparse(url).netloc)),
        'having_digits_in_domain': int(bool(re.search(r'\d', urlparse(url).netloc))),
        'number_of_digits_in_domain': len(re.findall(r'\d', urlparse(url).netloc)), 
        'having_repeated_digits_in_domain': int(bool(re.search(r'(\d)\1', urlparse(url).netloc))),
        'number_of_subdomains': urlparse(url).netloc.count('.') - 1,
        'having_dot_in_subdomain': int(bool(re.search(r'\.', urlparse(url).netloc.split('.')[0]))),
        'having_hyphen_in_subdomain': int(bool(re.search(r'-', urlparse(url).netloc.split('.')[0]))),
        'average_subdomain_length': sum(len(part) for part in urlparse(url).netloc.split('.')) / len(urlparse(url).netloc.split('.')),
        'average_number_of_dots_in_subdomain': urlparse(url).netloc.split('.').count('.') / len(urlparse(url).netloc.split('.')),
        'average_number_of_hyphens_in_subdomain': urlparse(url).netloc.split('.').count('-') / len(urlparse(url).netloc.split('.')),
        'having_special_characters_in_subdomain': int(bool(re.search(r'[^a-zA-Z0-9-]', urlparse(url).netloc.split('.')[0]))),
        'number_of_special_characters_in_subdomain': len(re.findall(r'[^a-zA-Z0-9-]', urlparse(url).netloc.split('.')[0])),
        'having_digits_in_subdomain': int(bool(re.search(r'\d', urlparse(url).netloc.split('.')[0]))),
        'number_of_digits_in_subdomain': len(re.findall(r'\d', urlparse(url).netloc.split('.')[0])),
        'having_repeated_digits_in_subdomain': int(bool(re.search(r'(\d)\1', urlparse(url).netloc.split('.')[0]))),
        'having_path': int(bool(urlparse(url).path)),
        'path_length': len(urlparse(url).path),
        'having_query': int(bool(urlparse(url).query)),
        'having_fragment': int(bool(urlparse(url).fragment)),
        'having_anchor': int(bool(urlparse(url).fragment)),
        'entropy_of_url': entropy(url),
        'entropy_of_domain': entropy(urlparse(url).netloc)
    }
    return data_dict

# Extract features
features = extract_features(url)

for key in features.keys():
    print(f"{key} : {features[key]}")
