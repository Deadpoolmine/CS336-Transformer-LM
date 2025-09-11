def round_up_to_power_of_2(a: int) -> int:
    if a <= 0:
        return 1
    if a == 1:
        return 2
    return 1 << (a - 1).bit_length()


def try_convert_to_bytes(x):
    if type(x) is bytes:
        return x
    else:
        return bytes([x])


def debug_print(verbose, *args):
    if verbose:
        print(*args)
