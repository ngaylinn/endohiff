"""Compute bitstring fitness with the HIFF function.
"""

import taichi as ti

from src.constants import BITSTR_POWER, BITSTR_LEN, BITSTR_DTYPE


@ti.func
def score_hiff(bitstr):
    # The one-bit substrings are "freebies" and automatically count towards the
    # hiff score. The HIFF score is an integer, but we use small float values
    # because that's convenient for calculating envioronmental fitness.
    hiff = ti.cast(BITSTR_LEN, ti.float16)

    # For all the powers of two up to BITSTR_LEN, look at all the substrings of
    # that size. We start with length 2, since we handled length 1 above.
    substr_len = 2
    mask = ti.cast(0b11, BITSTR_DTYPE)
    for p in range(BITSTR_POWER):
        # Use a bitmask as a sliding window, isolating each substring in turn.
        substr_mask = mask
        for s in range(BITSTR_LEN // substr_len):
            # Factor this substr into the final hiff score. It will contribute
            # substr_len to the score if and only if it is all 0's or all 1's.
            substr = bitstr & substr_mask
            score = substr_len * int(substr == 0 or substr == substr_mask)
            hiff += ti.cast(score, ti.float16)

            # Shift the mask to look at the next substr in order.
            # NOTE: The check here isn't necessary, but without it Taichi's
            # debugger will fire an overwhelming number of warnings.
            if s + 1 < BITSTR_LEN // substr_len:
                substr_mask <<= substr_len

        # Start looking at substr with twice the length as last iteration.
        # NOTE: The check here isn't necessary, but without it Taichi's
        # debugger will fire an overwhelming number of warnings.
        if p + 1 < BITSTR_POWER:
            mask = mask << substr_len | mask
            substr_len *= 2

    return hiff


@ti.kernel
def hiff_demo(bitstr: BITSTR_DTYPE):
    ti.loop_config(serialize=True)
    hiff = score_hiff(bitstr)
    print(f'HIFF: {hiff}')


# A demo to compute the weighted hiff score of some bit string.
if __name__ == '__main__':
    ti.init(ti.cpu, cpu_max_num_threads=1)
    bitstr = 0b0000000000000000000000000000000000000000000000000000000000000000
    hiff_demo(bitstr)

