import math

def gu32ops(bit_length, kernel):
    """
    compute the number of Giga uint32 operations required to multiply two numbers of the given bit length & kernel method
    """
    if kernel == "naive":
        return math.ceil(bit_length / 32)**2 / 1e9