from numpy.testing import assert_allclose

from reliability.Distributions import (
    Gumbel_Distribution,
)

atol = 1e-8
rtol = 1e-7


def test_Gumbel_Distribution():
    dist = Gumbel_Distribution(mu=15, sigma=2)
    assert_allclose(dist.mean, 13.845568670196934, rtol=rtol, atol=atol)
    assert_allclose(dist.standard_deviation, 2.565099660323728, rtol=rtol, atol=atol)
    assert_allclose(dist.variance, 6.579736267392906, rtol=rtol, atol=atol)
    assert_allclose(dist.skewness, -1.1395470994046486, rtol=rtol, atol=atol)
    assert_allclose(dist.kurtosis, 5.4, rtol=rtol, atol=atol)
    assert dist.param_title_long == "Gumbel Distribution (μ=15,σ=2)"
    assert_allclose(dist.quantile(0.2), 12.00012002648097, rtol=rtol, atol=atol)
    assert_allclose(dist.inverse_SF(q=0.7), 12.938139133682554, rtol=rtol, atol=atol)
    assert_allclose(dist.mean_residual_life(10), 4.349172610672009, rtol=rtol, atol=atol)
    xvals = [
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
            0.0004997499166249747,
            0.0049749162474832164,
            0.04741223204602183,
            0.11512925464970217,
            0.0230258509299404,
            0.003453877639491069,
        ],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.CDF(xvals=xvals, show_plot=True),
        [0.0009999999999999998, 0.010000000000000002, 0.09999999999999999, 0.9000000000000001, 0.99, 0.999],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.SF(xvals=xvals, show_plot=True),
        [0.999, 0.99, 0.9, 0.09999999999999984, 0.009999999999999969, 0.0010000000000000002],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.HF(xvals=xvals, show_plot=True),
        [
            0.0005002501667917664,
            0.0050251679267507236,
            0.05268025782891314,
            1.1512925464970236,
            2.3025850929940472,
            3.4538776394910684,
        ],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.CHF(xvals=xvals, show_plot=True),
        [
            0.0010005003335835344,
            0.01005033585350145,
            0.10536051565782628,
            2.3025850929940472,
            4.6051701859880945,
            6.907755278982137,
        ],
        rtol=rtol,
        atol=atol,
    )
