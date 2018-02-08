import random


def random_string(size, domain='abcdef1234567890'):
    """Creates a random string using provided set of symbols."""
    return ''.join([random.choice(domain) for _ in range(size)])
