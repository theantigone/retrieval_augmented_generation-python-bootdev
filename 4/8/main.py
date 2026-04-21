from lib import dot, euclidean_norm


def cosine_similarity(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError
    dot_product = dot(vec1, vec2)
    vec1_magnitude = euclidean_norm(vec1)
    vec2_magnitude = euclidean_norm(vec2)
    if vec1_magnitude == 0 or vec2_magnitude == 0:
        return 0.0
    return dot_product / (vec1_magnitude * vec2_magnitude)


pass
