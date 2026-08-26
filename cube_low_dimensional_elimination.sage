#!/usr/bin/env sage
"""
Exact certificate for the low-dimensional elimination in
cube_20260826PM0633.tex.

Run with SageMath 10.x:

    sage cube_low_dimensional_elimination.sage

The script performs two independent checks.

1. It enumerates the orbits of finite-edge subsets under the full cube
   automorphism group (S_2)^3 semidirect S_3.
2. For every orbit representative with e_+=3 or e_+=4, it constructs the
   normalized 6 by 6 Cartan matrix, expands all thirty-six 5 by 5 minors,
   and verifies the claimed triangular system in the localization relevant
   to the positive parameter domain.

All computations use exact rational-function and polynomial rings.
"""

from itertools import combinations, permutations, product
import sys

try:
    from sage.all import FractionField, Matrix, PolynomialRing, QQ
except ModuleNotFoundError:
    FractionField = Matrix = PolynomialRing = QQ = None


OPPOSITE_PAIRS = ((1, 2), (3, 4), (5, 6))
CUBE_EDGES = tuple(
    tuple(sorted((i, j)))
    for pair_a, pair_b in combinations(range(3), 2)
    for i in OPPOSITE_PAIRS[pair_a]
    for j in OPPOSITE_PAIRS[pair_b]
)

E3_CASES = (
    {"edges": ((1, 3), (1, 5), (3, 5)), "tree": ((1, 3), (1, 5)), "r": (3, 5)},
    {"edges": ((1, 3), (1, 5), (3, 6)), "tree": ((1, 3), (1, 5)), "r": (3, 6)},
    {"edges": ((1, 3), (1, 5), (4, 6)), "tree": ((1, 3), (1, 5)), "r": (4, 6)},
    {"edges": ((1, 3), (2, 5), (4, 6)), "tree": ((1, 3), (2, 5)), "r": (4, 6)},
)

E4_CASES = (
    {
        "edges": ((1, 3), (1, 4), (1, 5), (3, 5)),
        "tree": ((1, 3), (1, 5)),
        "q": (1, 4),
        "r": (3, 5),
    },
    {
        "edges": ((1, 3), (1, 4), (1, 5), (3, 6)),
        "tree": ((1, 3), (1, 5)),
        "q": (1, 4),
        "r": (3, 6),
    },
    {
        "edges": ((1, 3), (1, 4), (2, 5), (3, 5)),
        "tree": ((1, 3), (2, 5)),
        "q": (1, 4),
        "r": (3, 5),
    },
    {
        "edges": ((1, 3), (1, 4), (2, 5), (3, 6)),
        "tree": ((1, 3), (2, 5)),
        "q": (1, 4),
        "r": (3, 6),
    },
    {
        "edges": ((1, 3), (1, 5), (2, 4), (3, 5)),
        "tree": ((1, 3), (1, 5)),
        "q": (2, 4),
        "r": (3, 5),
    },
    {
        "edges": ((1, 3), (1, 5), (2, 4), (3, 6)),
        "tree": ((1, 3), (1, 5)),
        "q": (2, 4),
        "r": (3, 6),
    },
    {
        "edges": ((1, 3), (1, 5), (2, 4), (4, 5)),
        "tree": ((1, 3), (1, 5)),
        "q": (2, 4),
        "r": (4, 5),
    },
    {
        "edges": ((1, 3), (1, 5), (2, 4), (4, 6)),
        "tree": ((1, 3), (1, 5)),
        "q": (2, 4),
        "r": (4, 6),
    },
)


def cube_automorphisms():
    """Return the 48 permutations preserving the three opposite pairs."""
    automorphisms = []
    for pair_permutation in permutations(range(3)):
        for flips in product((0, 1), repeat=3):
            mapping = {}
            for source_pair in range(3):
                target_pair = pair_permutation[source_pair]
                for side in range(2):
                    mapping[OPPOSITE_PAIRS[source_pair][side]] = (
                        OPPOSITE_PAIRS[target_pair][side ^ flips[source_pair]]
                    )
            automorphisms.append(mapping)
    assert len(automorphisms) == 48
    return automorphisms


AUTOMORPHISMS = cube_automorphisms()


def act_on_edges(edge_set, mapping):
    return frozenset(
        tuple(sorted((mapping[i], mapping[j]))) for i, j in edge_set
    )


def pair_class(edge):
    first = next(k for k, pair in enumerate(OPPOSITE_PAIRS) if edge[0] in pair)
    second = next(k for k, pair in enumerate(OPPOSITE_PAIRS) if edge[1] in pair)
    return tuple(sorted((first, second)))


def fails_CO(edge_set):
    return len({pair_class(edge) for edge in edge_set}) == 3


def canonical_edge_set(edge_set):
    return min(
        tuple(sorted(act_on_edges(edge_set, mapping)))
        for mapping in AUTOMORPHISMS
    )


def verify_orbit_enumeration():
    for size, cases in ((3, E3_CASES), (4, E4_CASES)):
        all_representatives = {
            canonical_edge_set(frozenset(edge_set))
            for edge_set in combinations(CUBE_EDGES, size)
            if fails_CO(edge_set)
        }
        claimed_representatives = {
            canonical_edge_set(frozenset(case["edges"])) for case in cases
        }
        assert all_representatives == claimed_representatives
        assert len(all_representatives) == len(cases)
        print("orbit enumeration e_+={} : {} orbits".format(size, len(cases)))


def set_directed_pair(cartan, source, target, forward_entry, reverse_entry):
    cartan[source - 1, target - 1] = forward_entry
    cartan[target - 1, source - 1] = reverse_entry


def other_member(pair, member):
    return pair[1] if pair[0] == member else pair[0]


def normalized_cartan(
    coefficient_map,
    finite_edges,
    tree_edges,
    r_edge,
    mu1,
    mu2,
    mu3,
    r,
    q_edge=None,
    q=None,
):
    """
    Construct the normalized Cartan matrix used in the manuscript.

    The first opposite pair is oriented 2 -> 1.  The second and third
    opposite pairs point toward their respective endpoints of r_edge.
    Every displayed finite edge is oriented from its smaller endpoint to
    its larger endpoint.
    """
    base = r.parent().fraction_field()
    cartan = Matrix(base, 6, 6, 0)
    for i in range(6):
        cartan[i, i] = 2

    set_directed_pair(cartan, 2, 1, -1, -mu1)

    endpoint_two = next(vertex for vertex in r_edge if vertex in OPPOSITE_PAIRS[1])
    endpoint_three = next(vertex for vertex in r_edge if vertex in OPPOSITE_PAIRS[2])
    set_directed_pair(
        cartan,
        other_member(OPPOSITE_PAIRS[1], endpoint_two),
        endpoint_two,
        -1,
        -mu2,
    )
    set_directed_pair(
        cartan,
        other_member(OPPOSITE_PAIRS[2], endpoint_three),
        endpoint_three,
        -1,
        -mu3,
    )

    for edge in tree_edges:
        i, j = edge
        set_directed_pair(cartan, i, j, -1, -coefficient_map[edge])

    if q_edge is not None:
        i, j = q_edge
        set_directed_pair(
            cartan,
            i,
            j,
            -q,
            -coefficient_map[q_edge] / q,
        )

    i, j = r_edge
    set_directed_pair(
        cartan,
        i,
        j,
        -r,
        -coefficient_map[r_edge] / r,
    )

    assert set(finite_edges) == set(tree_edges) | {r_edge} | (
        {q_edge} if q_edge is not None else set()
    )
    return cartan


def cleared_five_by_five_minors(cartan, polynomial_ring):
    """Return the nonzero numerators of all thirty-six maximal minors."""
    minors = []
    for row_set in combinations(range(6), 5):
        for column_set in combinations(range(6), 5):
            determinant = cartan.matrix_from_rows_and_columns(
                row_set, column_set
            ).det()
            numerator = polynomial_ring(determinant.numerator())
            if numerator:
                minors.append(numerator)
    return minors


def localized_membership(polynomial, groebner_basis, localization_element):
    """
    Test membership after inverting localization_element.

    The determinant denominators contain only a bounded power of r; the
    generous upper bound 12 is well above the largest possible power in a
    5 by 5 minor.
    """
    for exponent in range(13):
        candidate = localization_element ** exponent * polynomial
        if candidate.reduce(groebner_basis) == 0:
            return True
    return False


def verify_localized_ideal(minors, triangular_generators, r, label):
    ring = r.parent()
    minor_ideal = ring.ideal(minors)
    triangular_ideal = ring.ideal(triangular_generators)
    minor_basis = list(minor_ideal.groebner_basis())
    triangular_basis = list(triangular_ideal.groebner_basis())

    for generator in minors:
        assert localized_membership(generator, triangular_basis, r)
    for generator in triangular_generators:
        assert localized_membership(generator, minor_basis, r)
    print(label + " : localized ideals agree")


def verify_e3_elimination():
    coefficient_ring = PolynomialRing(QQ, names=("a0", "a1", "a2"))
    a0, a1, a2 = coefficient_ring.gens()
    coefficient_field = FractionField(coefficient_ring)
    ring = PolynomialRing(
        coefficient_field,
        names=("mu1", "mu2", "mu3", "r"),
        order="lex",
    )
    mu1, mu2, mu3, r = ring.gens()

    triangular_systems = (
        (
            a2 * mu1 - 4 * a2 - 2 * a1 * r,
            mu2 - 4 - 2 * r,
            a0 * mu3 - 4 * a0 - 2 * a1 * r,
            a1 * r ** 2 - a0 * a2,
        ),
        (
            a2 * mu1 - 4 * a2 - 4 * a1 * r,
            mu2 - 4 - 4 * r,
            a0 * mu3 - 4 * a0 - a1 * a2 - 4 * a1 * r,
            4 * a1 * r ** 2 + a1 * a2 * r - a0 * a2,
        ),
        (
            a2 * mu1 - (4 - a0) * a2 - 2 * a1 * r,
            mu2 - 4 - 2 * r,
            a0 * mu3 - 4 * a0 - (a1 - a0) * a2 - 2 * a1 * r,
            2 * a1 * r ** 2 - a2 * (a0 - a1) * r - 2 * a0 * a2,
        ),
        (
            a2 * mu1 - (4 - a0) * a2 - a1 * r,
            mu2 - 4 + a0 - r,
            (
                a0 * mu3
                - 4 * a0
                + a0 * a1
                + a0 * a2
                - a1 * a2
                - a1 * r
            ),
            (
                a1 * r ** 2
                + (-a0 * a1 - a0 * a2 + a1 * a2) * r
                + a0 * a2 * (a0 - 4)
            ),
        ),
    )

    for index, (case, triangular_system) in enumerate(
        zip(E3_CASES, triangular_systems), start=1
    ):
        coefficient_map = {
            edge: value
            for edge, value in zip(sorted(case["edges"]), (a0, a1, a2))
        }
        cartan = normalized_cartan(
            coefficient_map,
            case["edges"],
            case["tree"],
            case["r"],
            mu1,
            mu2,
            mu3,
            r,
        )
        minors = cleared_five_by_five_minors(cartan, ring)
        verify_localized_ideal(
            minors,
            [ring(polynomial) for polynomial in triangular_system],
            r,
            "e_+=3 row {}".format(index),
        )


def e4_triangular_system(index, a0, a1, a2, a3, q, mu1, mu2, mu3, r):
    d = a0 * q - 1
    e = (a0 - 4) * q - 1
    f = a0 * a2 * q - 4 * a0 * q - a2
    h = (a0 + a1 - 4) * q - 1
    w_aux = 2 * a1 * h * r + a3 * (
        a0 * (4 - a0) * q + a0 - a1
    )

    linear_coefficients = (
        (a3, 1, a0 * q + 2 * a1),
        (a3, 1, (q + 2) * (a0 * q + 2 * a1)),
        (2 * a3, a1 + 2 * q, a0 * q + 2 * a1),
        (a3, a1 + 2 * q, (q + 2) * (a0 * q + 2 * a1)),
        (a3, 1, a0 * q + a2),
        (a3, 1, (q + 1) * (a0 * q + a2)),
        (a3 * d, d, f),
        (a3 * d, d, e * f),
    )

    right_hand_sides = (
        (
            a3 * (4 - a1) + a2 * (q + 2) * r,
            4 + (q + 2) * r,
            4 * (a0 * q + 2 * a1) + 2 * (a2 * q * r - a1 * a3),
        ),
        (
            a3 * (4 - a1) + 2 * a2 * (q + 2) * r,
            4 + 2 * (q + 2) * r,
            (
                4 * (q + 2) * (a0 * q + 2 * a1)
                - 2 * a1 * a3 * (q + 2)
                + 2 * a2 * a3 * q
                + 4 * a2 * q * (q + 2) * r
            ),
        ),
        (
            2 * a3 * (4 - a1) + a2 * (q + 2) * r,
            q * ((q + 2) * (r - a0) + 8 - 2 * a1),
            (4 - a2) * (a0 * q + 2 * a1) - 2 * a1 * a3 + a2 * q * r,
        ),
        (
            a3 * (4 - a1) + a2 * (q + 2) * r,
            q * (2 * (q + 2) * r - a0 * (q + 2) + 8 - 2 * a1),
            (
                4 * (q + 2) * (a0 * q + 2 * a1)
                - a0 * a2 * q * (q + 2)
                + a1 * a2 * a3
                - 2 * a1 * (a2 + a3) * (q + 2)
                + 2 * a2 * a3 * q
                + 2 * a2 * q * (q + 2) * r
            ),
        ),
        (
            a3 * (4 - a2) + a1 * (a0 * q + a2) + 2 * a1 * (q + 1) * r,
            4 + a0 * q + 2 * (q + 1) * r,
            4 * (a0 * q + a2) + 2 * a1 * q * r - a2 * a3,
        ),
        (
            (
                a3 * (4 - a2)
                + a1 * (a0 * q + a2)
                + a1 * r * ((4 - a1) * q + 4)
            ),
            4 + a0 * q + r * ((4 - a1) * q + 4),
            (
                4 * (q + 1) * (a0 * q + a2)
                + a1 * q * r * ((4 - a1) * q + 4)
                + a3 * (a1 * q - a2 * (q + 1))
            ),
        ),
        (
            a3 * (4 - a0) * d + a1 * (a2 * d + r * e - 4 * a0 * q),
            r * e - 4,
            4 * (f + a0 * a3 * q - a1 * q * r),
        ),
        (
            a3 * (4 - a0) * d + a1 * (a2 * d + 2 * r * h - 4 * a0 * q),
            2 * (r * h - 2),
            4 * e * f - 4 * q * w_aux,
        ),
    )

    leading_coefficients = (
        a2 * q * (q + 2),
        2 * a2 * q * (q + 2),
        a2 * q * (q + 2),
        2 * a2 * q * (q + 2),
        2 * a1 * q * (q + 1),
        a1 * q * ((a1 - 4) * q - 4),
        a1 * q * e,
        4 * a1 * q * h,
    )

    middle_coefficients = (
        -a1 * a3 * (q + 2),
        -a3 * (a1 * (q + 2) - a2 * q),
        -(q + 2) * (a0 * a2 * q + 2 * a1 * a2 + 2 * a1 * a3),
        (
            -a0 * a2 * q * (q + 2)
            + a1 * a2 * a3
            - 2 * a1 * (a2 + a3) * (q + 2)
            + 2 * a2 * a3 * q
        ),
        a1 * q * (a0 * q + a2) - a2 * a3 * (q + 1),
        -a1 * q * (a0 * q + a2 + a3) + a2 * a3 * (q + 1),
        -q * (a0 * a3 * e + a1 * (4 * a0 * q - a2 * d)),
        -2 * q * (a0 * a3 * e + a1 * (4 * a0 * q - a2 * d + a3)),
    )

    constant_coefficients = (
        -2 * a3 * (a0 * q + 2 * a1),
        -a3 * (a0 * q + 2 * a1),
        2 * a3 * (a1 - 4) * (a0 * q + 2 * a1),
        a3 * (a1 - 4) * (a0 * q + 2 * a1),
        -2 * a3 * (a0 * q + a2),
        a3 * (a0 * q + a2),
        a3 * (a0 * q * (4 - a2) + a2),
        a3 * (a0 * q * (4 - a2) + a2),
    )

    row = index - 1
    l1, l2, l3 = linear_coefficients[row]
    r1, r2, r3 = right_hand_sides[row]
    quadratic = (
        leading_coefficients[row] * r ** 2
        + middle_coefficients[row] * r
        + constant_coefficients[row]
    )
    return (
        l1 * mu1 - r1,
        l2 * mu2 - r2,
        l3 * mu3 - r3,
        quadratic,
    )


def verify_e4_elimination():
    coefficient_ring = PolynomialRing(
        QQ, names=("a0", "a1", "a2", "a3", "q")
    )
    a0, a1, a2, a3, q = coefficient_ring.gens()
    coefficient_field = FractionField(coefficient_ring)
    ring = PolynomialRing(
        coefficient_field,
        names=("mu1", "mu2", "mu3", "r"),
        order="lex",
    )
    mu1, mu2, mu3, r = ring.gens()

    for index, case in enumerate(E4_CASES, start=1):
        coefficient_map = {
            edge: value
            for edge, value in zip(
                sorted(case["edges"]), (a0, a1, a2, a3)
            )
        }
        cartan = normalized_cartan(
            coefficient_map,
            case["edges"],
            case["tree"],
            case["r"],
            mu1,
            mu2,
            mu3,
            r,
            q_edge=case["q"],
            q=q,
        )
        minors = cleared_five_by_five_minors(cartan, ring)
        triangular_system = [
            ring(polynomial)
            for polynomial in e4_triangular_system(
                index, a0, a1, a2, a3, q, mu1, mu2, mu3, r
            )
        ]
        verify_localized_ideal(
            minors,
            triangular_system,
            r,
            "e_+=4 row {}".format(index),
        )


def main():
    verify_orbit_enumeration()
    if "--orbits-only" in sys.argv:
        print("pure-Python orbit certificate verified")
        return
    if PolynomialRing is None:
        raise RuntimeError(
            "SageMath is required for the exact elimination; "
            "use --orbits-only for the pure-Python orbit check"
        )
    verify_e3_elimination()
    verify_e4_elimination()
    print("all exact low-dimensional certificates verified")


if __name__ == "__main__":
    main()
