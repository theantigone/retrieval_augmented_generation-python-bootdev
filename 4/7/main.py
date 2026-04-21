def dot(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("not the same length")
    sum = 0
    for i in range(len(vec1)):
        sum += vec1[i] * vec2[i]
    return sum


pass
