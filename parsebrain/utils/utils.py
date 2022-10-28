from itertools import islice, repeat, chain


def pad(x, padding_value=-1):
    """
    >>> x = [[1,2,3],[1],[],[1,2,3,4]]
    >>> pad(x,padding_value=0)
    [[1, 2, 3, 0], [1, 0, 0, 0], [0, 0, 0, 0], [1, 2, 3, 4]]
    """
    zeros = repeat(padding_value)
    n = max(map(len, x))
    return [list(islice(chain(row, zeros), n)) for row in x]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
