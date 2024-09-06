from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose

from reliability.Datasets import mileage
from reliability.Distributions import Normal_Distribution
from reliability.Reliability_testing import (
    KStest,
    chi2test,
    likelihood_plot,
    one_sample_proportion,
    reliability_test_duration,
    reliability_test_planner,
    sample_size_no_failures,
    sequential_sampling_chart,
    two_proportion_test,
)

atol = 1e-8
rtol = 1e-7


def test_reliability_one_sample_proportion():
    results = one_sample_proportion(trials=30, successes=29)
    assert_allclose(results[0], 0.8278305443665873, rtol=rtol, atol=atol)
    assert_allclose(results[1], 0.9991564290733695, rtol=rtol, atol=atol)


def test_two_proportion_test():
    results = two_proportion_test(
        sample_1_trials=500,
        sample_1_successes=490,
        sample_2_trials=800,
        sample_2_successes=770,
    )
    assert_allclose(results[0], -0.0004972498915250083, rtol=rtol, atol=atol)
    assert_allclose(results[1], 0.03549724989152493, rtol=rtol, atol=atol)


def test_sample_size_no_faillures():
    sample_size: int = sample_size_no_failures(reliability=0.999)
    assert sample_size == 2995


def test_reliability_test_planner():
    test_plan = reliability_test_planner(test_duration=19520, CI=0.8, number_of_failures=7, one_sided=False)
    assert_allclose(test_plan.CI, 0.8, rtol=rtol, atol=atol)
    assert_allclose(test_plan.MTBF, 1658.3248534993454, rtol=rtol, atol=atol)
    assert_allclose(test_plan.number_of_failures, 7, rtol=rtol, atol=atol)
    assert_allclose(test_plan.test_duration, 19520, rtol=rtol, atol=atol)

    # Test Case 2 for supplying MTBF
    mtbf = reliability_test_planner(test_duration=19520, CI=0.8, MTBF=1660, one_sided=False)
    assert_allclose(mtbf.number_of_failures, 6, rtol=rtol, atol=atol)


def test_reliability_test_duration():
    results = reliability_test_duration(
        MTBF_required=2500,
        MTBF_design=3000,
        consumer_risk=0.2,
        producer_risk=0.2,
    )
    assert_allclose(results.duration_solution, 231615.79491309822, rtol=rtol, atol=atol)
    results.print_results()
    results.plot()
    plt.close()


def test_chi2test():
    data = np.asarray(mileage().failures)
    dist = Normal_Distribution(mu=30011.0, sigma=10472)
    bins = [
        0,
        13417,
        18104,
        22791,
        27478,
        32165,
        36852,
        41539,
        46226,
        np.inf,
    ]  # it is not necessary to specify the bins and leaving them unspecified is usually best
    chisq = chi2test(distribution=dist, data=data, bins=bins)
    chisq.print_results()
    chisq.plot()
    assert_allclose(chisq.chisquared_critical_value, 12.591587243743977, rtol=rtol, atol=atol)
    assert_allclose(chisq.chisquared_statistic, 3.1294947845652, rtol=rtol, atol=atol)
    assert chisq.hypothesis == "ACCEPT"


def test_KStest():
    data = mileage().failures
    dist = Normal_Distribution(mu=30011.0, sigma=10472)
    results = KStest(distribution=dist, data=data)
    results.print_results()
    results.plot()
    assert_allclose(results.KS_critical_value, 0.13402791648569978, rtol=rtol, atol=atol)
    assert_allclose(results.KS_statistic, 0.07162465859560846, rtol=rtol, atol=atol)
    assert results.hypothesis == "ACCEPT"


dist_types: list[str] = ["Weibull", "Gamma", "Loglogistic", "Normal", "Lognormal", "Gumbel"]


@pytest.mark.parametrize("dist_type", dist_types)
def test_likelihood_plot(dist_type: str):
    old_design = [
        2,
        9,
        23,
        38,
        67,
        2,
        11,
        28,
        40,
        76,
        3,
        17,
        33,
        45,
        90,
        4,
        17,
        34,
        55,
        115,
        6,
        19,
        34,
        56,
        126,
        9,
        21,
        37,
        57,
        197,
    ]
    new_design = [15, 116, 32, 148, 61, 178, 67, 181, 75, 183]

    fig = likelihood_plot(distribution=dist_type, failures=old_design, CI=[0.9, 0.95])
    assert isinstance(fig, plt.Figure)

    fig = likelihood_plot(distribution=dist_type, failures=new_design, CI=[0.9, 0.95])
    assert isinstance(fig, plt.Figure)


def test_sequential_sampling():
    test_results = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
    ]
    results = sequential_sampling_chart(p1=0.01, p2=0.10, alpha=0.05, beta=0.10, test_results=test_results)
    assert results.shape[0] == 101
    assert results.shape[1] == 3
