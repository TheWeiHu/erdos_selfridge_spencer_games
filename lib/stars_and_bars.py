"""Stars and Bars Algorithm

Blatantly taken from:
https://gist.github.com/thebertster/0362ae4c8197f0d6bee10d484b509912

Generates all possible ways to  put n indistinguishable balls into k distinguishable
bins, for any given n and k.

"""


def stars_and_bars(bins, stars, allow_empty=True):
    """
    Non-recursive generator that returns the possible ways that n indistinguishable
    objects can be distributed between k distinguishible bins (allowing empty bins).

    Total number of arrangements = (n+k-1)! / n!(k-1)! if empty bins are allowed
    Total number of arrangements = (n-1)! / (n-k)!(k-1)! if empty bins are not allowed

    Parameters
    ----------
    bins : int
        Number of distinguishible bins (k)
    stars : int
        Number of indistinguishible objects (n)
    allow_empty : boolean
        If True, empty bins are allowed; default is True
    """

    if bins < 1 or stars < 1:
        raise ValueError(
            "Number of stars and bins must both be greater than or equal to 1."
        )
    if not allow_empty and stars < bins:
        raise ValueError(
            "Number of stars must be greater than or equal to the number of bins."
        )

    # If there is only one bin, there is only one arrangement!
    if bins == 1:
        yield stars
        return

    # If empty bins are not allowed, distribute (star-bins) stars and add an extra star
    # to each bin when yielding.
    if not allow_empty:
        if stars == bins:
            # If same number of stars and bins, then there is only one arrangement!
            yield tuple([1] * bins)
            return
        stars -= bins

    # 'bars' holds the queue or stack of positions of the bars in the stars and bars
    # arrangement (including a bar at the beginning and end) and the level of iteration
    # that this stack item has reached.
    # Initial stack holds a single arrangement ||...||*****...****|
    # with an iteration level of 1.
    bars = [([0] * bins + [stars], 1)]

    # Generates arrangements in lexically ascending order.

    # Iterate through the current queue of arrangements until no more are left (all
    # arrangements have been yielded).
    while bars:
        new_bars = []

        for elem in bars:
            # Iterate through inner arrangements of b, yielding each arrangement and
            # queuing each arrangement for further iteration except the very first.
            for arr in range(elem[0][-2], stars + 1):
                new_bar = elem[0][1:bins] + [arr, stars]
                if elem[1] < bins - 1 and arr > 0:
                    new_bars.append((new_bar, elem[1] + 1))

                # Translate the stars and bars into a tuple.
                yield tuple(
                    new_bar[y] - new_bar[y - 1] + (0 if allow_empty else 1)
                    for y in range(1, bins + 1)
                )

        bars = new_bars


def main():
    """  Examines the order of arrangements yielded by the generator.
    """
    print(list(stars_and_bars(4, 15)))


if __name__ == "__main__":
    main()
