""" Bucket Partition Algorithm

Given n identical tokens, and k buckets, partition the tokens into the buckets such that
the number of ways to choose different subsets is maximized.

The algorithm returns the possible number of different subsets in the maximal case.

There exists a correspondence between this maximal value and the maximum size of the
attacker action space of an Erdos-Selfridge-Spencer game, which n being the number of
pieces and k being the number of levels.

Todo:
    * Consider adding type hints.
"""

import math


def maximum_choices(n_tokens, n_buckets):
    """ Given n identical tokens, and k buckets, partition the tokens into the buckets
    such that the number of ways to choose different subsets is maximized.

    Args:
        n_tokens: the number of identical tokens to be put into buckets.
        n_buckets: the number of buckets available (not necessarily all used).
    Returns:
        the possible number of different subsets in the maximal case.
    """
    # We observe thatt the number of tokens in each bucket will either be
    # math.ceil(n_tokens / n_buckets) or math.floor(n_tokens / n_buckets)
    # Proof:
    # https://math.stackexchange.com/questions/1924116/maximizing-product-given-a-constraint-on-sum
    # With that constraint, we just have to calculate how many buckets have the floor
    # value vs. the ceil value.
    floor_count = n_buckets * (math.ceil(n_tokens / n_buckets)) - n_tokens
    ceil_count = n_buckets - floor_count
    # Apply product rule to get the maximum number of choices.
    return (math.ceil(n_tokens / n_buckets) + 1) ** ceil_count * (
        math.floor(n_tokens / n_buckets) + 1
    ) ** floor_count


def main():
    """Tests implemented algorithms, verifying that the maximum number of choices
    increases as the number of buckets increases.
    """
    print(maximum_choices(15, 1000))
    print(maximum_choices(15, 999))
    print(maximum_choices(15, 9))
    print(maximum_choices(15, 8))
    print(maximum_choices(15, 7))
    print(maximum_choices(15, 6))
    print(maximum_choices(15, 5))
    print(maximum_choices(15, 4))
    print(maximum_choices(15, 3))
    print(maximum_choices(15, 2))
    print(maximum_choices(15, 1))


if __name__ == "__main__":
    main()
