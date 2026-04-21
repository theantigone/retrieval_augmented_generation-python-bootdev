def add_vectors(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("not equal")
    sums = []
    for i in range(len(vec1)):
        sums.append(vec1[i] + vec2[i])
    return sums


def subtract_vectors(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("not equal")
    sums = []
    for i in range(len(vec1)):
        sums.append(vec1[i] - vec2[i])
    return sums
