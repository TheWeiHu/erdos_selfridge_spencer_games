"""Stars and Bars Algorithm

Blatantly taken from:
https://gist.github.com/thebertster/0362ae4c8197f0d6bee10d484b509912

Generates all possible ways to  put n indistinguishable balls into k distinguishable
bins, for any given n and k.

Todo:
    * Make changes to original code to follow best Python practice.

"""


def stars_and_bars(bins, stars, reverse=False, allow_empty=True):
    """
    Non-recursive generator that returns the possible ways that n indistinguishable objects
    can be distributed between k distinguishible bins (allowing empty bins)

    Total number of arrangements = (n+k-1)! / n!(k-1)! if empty bins are allowed
    Total number of arrangements = (n-1)! / (n-k)!(k-1)! if empty bins are not allowed

    Parameters
    ----------
    bins : int
        Number of distinguishible bins (k)
    stars : int
        Number of indistinguishible objects (n)
    reverse : boolean
        If True, generator produces a reverse-order sequence; default is False
    allow_empty : boolean
        If True, empty bins are allowed; default is True
    """

    if bins < 1 or stars < 1:
        raise ValueError(
            "Number of objects (stars) and bins must both be greater than or equal to 1."
        )
    if not allow_empty and stars < bins:
        raise ValueError(
            "Number of objects (stars) must be greater than or equal to the number of bins."
        )

    # If there is only one bin, there is only one arrangement!
    if bins == 1:
        yield stars,
        return

    # If empty bins are not allowed, distribute (star-bins) stars and add an extra star to each bin when yielding.
    if not allow_empty:
        if stars == bins:
            # If same number of stars and bins, then there is only one arrangement!
            yield tuple([1] * bins)
            return
        else:
            stars -= bins

    # 'bars' holds the queue or stack of positions of the bars in the stars and bars arrangement
    # (including a bar at the beginning and end) and the level of iteration that this stack item has reached.
    # Initial stack holds a single arrangement ||...||*****...****| with an iteration level of 1.
    bars = [([0] * bins + [stars], 1)]

    if reverse:
        # Generates arrangements in lexically descending order.

        # Iterate through the current stack of arrangement until no more are left (all arrangements have been yielded).
        while len(bars) > 0:
            # Pop the top-most arrangement off the stack for processing.
            b = bars.pop()

            # If the arrangement's iteration level indicates this position has been fully "explored" (i.e. the
            # arrangement is the result of a full iteration over the number of bins), then we can yield a result.
            if b[1] == bins:
                # Translate the stars and bars into a tuple giving the number of objects in each bin for that arrangement.
                # Because of the static placement of bars at the beginning and end, this is therefore a simple matter of
                # taking the difference between each pair of bar positions.
                # e.g. (0, 2 ,5 ,6 ,12) -> |**|***|*|******| -> (2, 3, 1, 6)
                yield tuple(
                    b[0][y] - b[0][y - 1] + (0 if allow_empty else 1)
                    for y in range(1, bins + 1)
                )

            # If the popped arrangement is not fully iterated, add to the stack all arrangements where the first b[1] bars are
            # fixed but move the remaining bars incrementally to the right and indicate that each arrangement is one level
            # deeper in the iteration.
            else:
                bar = b[0][: b[1]]
                for x in range(b[0][b[1]], stars + 1):
                    newBar = bar + [x] * (bins - b[1]) + [stars]
                    bars.append((newBar, b[1] + 1))

    else:
        # Generates arrangements in lexically ascending order.

        # Iterate through the current queue of arrangements until no more are left (all arrangements have been yielded).
        while len(bars) > 0:
            newBars = []

            for b in bars:
                # Iterate through inner arrangements of b, yielding each arrangement and queuing each
                # arrangement for further iteration except the very first
                for x in range(b[0][-2], stars + 1):
                    newBar = b[0][1:bins] + [x, stars]
                    if b[1] < bins - 1 and x > 0:
                        newBars.append((newBar, b[1] + 1))

                    # Translate the stars and bars into a tuple
                    yield tuple(
                        newBar[y] - newBar[y - 1] + (0 if allow_empty else 1)
                        for y in range(1, bins + 1)
                    )

            bars = newBars
