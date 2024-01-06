from numpy.testing import assert_allclose

from reliability.Reliability_testing import one_sample_proportion, sample_size_no_failures, two_proportion_test

atol = 1e-8
rtol = 1e-7


def test_reliability_one_sample_proportion():
    results = one_sample_proportion(trials=30, successes=29)
    assert_allclose(results[0], 0.8278305443665873, rtol=rtol, atol=atol)
    assert_allclose(results[1], 0.9991564290733695, rtol=rtol, atol=atol)


def test_two_proportion_test():
    results = two_proportion_test(
        sample_1_trials=500, sample_1_successes=490, sample_2_trials=800, sample_2_successes=770
    )
    assert_allclose(results[0], -0.0004972498915250083, rtol=rtol, atol=atol)
    assert_allclose(results[1], 0.03549724989152493, rtol=rtol, atol=atol)


def test_sample_size_no_faillures():
    sample_size = sample_size_no_failures(reliability=0.999)
    assert sample_size == 2995
