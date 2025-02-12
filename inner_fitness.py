"""Compute metrics on the inner population, like hiff score and one count.
"""

import taichi as ti

from constants import BITSTR_POWER, BITSTR_LEN, BITSTR_DTYPE, NUM_WEIGHTS


@ti.func
def weighted_hiff(bitstr, weights):
    # The one-bit substrings are "freebies" and automatically count towards the
    # hiff score. Don't use weights for these, since that would be pointless.
    hiff = ti.cast(BITSTR_LEN, ti.int16)
    fitness = ti.cast(BITSTR_LEN, ti.float32)

    # For all the powers of two up to BITSTR_LEN, look at all the substrings of
    # that size. We start with length 2, since we handled length 1 above.
    substr_len = 2
    mask = ti.cast(0b11, BITSTR_DTYPE)
    w = 0
    for p in range(BITSTR_POWER):
        # Use a bitmask as a sliding window, isolating each substring in turn.
        substr_mask = mask
        for s in range(BITSTR_LEN // substr_len):
            # Factor this substr into the final hiff score. It will contribute
            # substr_len to the hiff score (adjusted by weight) if and only if
            # it is all 0's or all 1's.
            substr = bitstr & substr_mask
            score = substr_len * int(substr == 0 or substr == substr_mask)
            fitness += score * weights[w] + 1.0 * (1.0 - weights[w])
            hiff += ti.cast(score, ti.int16)

            # Shift the mask to look at the next substr in order.
            # NOTE: The check here isn't necessary, but without it Taichi's
            # debugger will fire an overwhelming number of warnings.
            if s + 1 < BITSTR_LEN // substr_len:
                substr_mask <<= substr_len
                w += 1

        # Start looking at substr with twice the length as last iteration.
        # NOTE: The check here isn't necessary, but without it Taichi's
        # debugger will fire an overwhelming number of warnings.
        if p + 1 < BITSTR_POWER:
            mask = mask << substr_len | mask
            substr_len *= 2

    return fitness, hiff


@ti.func
def hiff(bitstr):
    fitness, hiff = weighted_hiff(bitstr, ti.Vector([1.0] * NUM_WEIGHTS))
    return hiff


@ti.func
def count_ones(bitstr):
    one_count = ti.cast(0, ti.int8)
    for b in range(BITSTR_LEN):
        one_count += ti.cast((bitstr >> b) & 1, ti.int8)
    return one_count


@ti.kernel
def hiff_demo(bitstr: BITSTR_DTYPE, weights: ti.template()):
    ti.loop_config(serialize=True)
    fitness, hiff = weighted_hiff(bitstr, weights)
    print(f'Fitness: {fitness:0.2f}; HIFF: {hiff}')


# A demo to compute the weighted hiff score of some bit string.
if __name__ == '__main__':
    ti.init(ti.cpu, cpu_max_num_threads=1)
    bitstr = 0b0000000000000000000000000000000000000000000000000000000000000000
    weights = ti.Vector([1.0] * NUM_WEIGHTS)
    hiff_demo(bitstr, weights)
