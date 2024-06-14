import pytest
from numpy.testing import assert_allclose

from reliability.Datasets import automotive
from reliability.Nonparametric import KaplanMeier, NelsonAalen, RankAdjustment

failures = automotive().failures
right_censored = automotive().right_censored

atol = 1e-8
rtol = 1e-7


def test_KaplanMeier():
    KMF = KaplanMeier(failures=failures, right_censored=right_censored)
    KMF.print_results()
    assert_allclose(sum(KMF.KM), 22.999467075124368, rtol=rtol, atol=atol)
    assert_allclose(sum(KMF.CDF), 9.999999999999998, rtol=rtol, atol=atol)
    assert_allclose(sum(KMF.SF_lower), 25.645200968606833, rtol=rtol, atol=atol)
    assert_allclose(sum(KMF.SF_upper), 38.377898733187926, rtol=rtol, atol=atol)


def test_NelsonAalen():
    NAF = NelsonAalen(failures=failures, right_censored=right_censored)
    NAF.print_results()
    assert_allclose(sum(NAF.NA), 23.29546519223915, rtol=rtol, atol=atol)
    assert_allclose(sum(NAF.CDF), 9.65601230386857, rtol=rtol, atol=atol)
    assert_allclose(sum(NAF.SF_lower), 25.78585422689871, rtol=rtol, atol=atol)
    assert_allclose(sum(NAF.SF_upper), 38.97276923275063, rtol=rtol, atol=atol)


def test_RankAdjustment():
    RAF = RankAdjustment(failures=failures, right_censored=right_censored)
    RAF.print_results()
    assert_allclose(sum(RAF.RA), 23.654828586003394, rtol=rtol, atol=atol)
    assert_allclose(sum(RAF.CDF), 4.965708127420795, rtol=rtol, atol=atol)
    assert_allclose(sum(RAF.SF_lower), 13.347349754628674, rtol=rtol, atol=atol)
    assert_allclose(sum(RAF.SF_upper), 20.858236210538312, rtol=rtol, atol=atol)


plot_tyes = ["CDF", "SF", "CHF"]
bool_list = [True, False]


@pytest.mark.parametrize("plot_type", plot_tyes)
@pytest.mark.parametrize("plot_CI", bool_list)
def test_KaplanMeier_plot(plot_type, plot_CI):
    KMF = KaplanMeier(failures=failures, right_censored=right_censored)
    KMF.plot(plot_type=plot_type, plot_CI=plot_CI)


@pytest.mark.parametrize("plot_type", plot_tyes)
@pytest.mark.parametrize("plot_CI", bool_list)
def test_NelsonAalen_plot(plot_type, plot_CI):
    NAF = NelsonAalen(failures=failures, right_censored=right_censored)
    NAF.plot(plot_type=plot_type, plot_CI=plot_CI)


@pytest.mark.parametrize("plot_type", plot_tyes)
@pytest.mark.parametrize("plot_CI", bool_list)
def test_RankAdjustment_plot(plot_type, plot_CI):
    RAF = RankAdjustment(failures=failures, right_censored=right_censored)
    RAF.plot(plot_type=plot_type, plot_CI=plot_CI)
