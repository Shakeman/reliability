from numpy.testing import assert_allclose

from reliability.Distributions import (
    Gamma_Distribution,
)

atol = 1e-8
rtol = 1e-7


def test_Gamma_Distribution():
    dist = Gamma_Distribution(alpha=5, beta=2, gamma=10)
    assert_allclose(dist.mean, 20, rtol=rtol, atol=atol)
    assert_allclose(dist.standard_deviation, 7.0710678118654755, rtol=rtol, atol=atol)
    assert_allclose(dist.variance, 50, rtol=rtol, atol=atol)
    assert_allclose(dist.skewness, 1.414213562373095, rtol=rtol, atol=atol)
    assert_allclose(dist.kurtosis, 6, rtol=rtol, atol=atol)
    assert dist.param_title_long == "Gamma Distribution (α=5,β=2,γ=10)"
    assert_allclose(dist.quantile(0.2), 14.121941545164923, rtol=rtol, atol=atol)
    assert_allclose(dist.inverse_SF(q=0.7), 15.486746053517457, rtol=rtol, atol=atol)
    assert_allclose(dist.mean_residual_life(20), 6.666666666666647, rtol=rtol, atol=atol)
    xvals = [
        dist.gamma - 1,
        dist.quantile(0.001),
        dist.quantile(0.01),
        dist.quantile(0.1),
        dist.quantile(0.9),
        dist.quantile(0.99),
        dist.quantile(0.999),
    ]
    assert_allclose(
        dist.PDF(xvals=xvals, show_plot=True),
        [
            0.0,
            0.008677353779839614,
            0.02560943552734864,
            0.06249207734544239,
            0.015909786387521992,
            0.001738163417685293,
            0.00018045617911753266,
        ],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.CDF(xvals=xvals, show_plot=True),
        [0.0, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.SF(xvals=xvals, show_plot=True),
        [1.0, 0.999, 0.99, 0.9, 0.1, 0.01, 0.001],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.HF(xvals=xvals, show_plot=True),
        [
            0.0,
            0.008686039819659272,
            0.025868116694291555,
            0.06943564149493599,
            0.15909786387522004,
            0.17381634176852898,
            0.18045617911753245,
        ],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.CHF(xvals=xvals, show_plot=True),
        [
            -0.0,
            0.0010005003335835344,
            0.01005033585350145,
            0.10536051565782628,
            2.3025850929940463,
            4.605170185988089,
            6.907755278982136,
        ],
        rtol=rtol,
        atol=atol,
    )
