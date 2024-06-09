from numpy.testing import assert_allclose

from reliability.Distributions import (
    Competing_Risks_Model,
    Mixture_Model,
    Normal_Distribution,
    Weibull_Distribution,
)

atol = 1e-8
rtol = 1e-7


def test_Competing_Risks_Model():
    distributions = [Weibull_Distribution(alpha=30, beta=2), Normal_Distribution(mu=35, sigma=5)]
    dist = Competing_Risks_Model(distributions=distributions)
    assert_allclose(dist.mean, 23.707625152181073, rtol=rtol, atol=atol)
    assert_allclose(dist.standard_deviation, 9.832880925543204, rtol=rtol, atol=atol)
    assert_allclose(dist.variance, 96.68554729591138, rtol=rtol, atol=atol)
    assert_allclose(dist.skewness, -0.20597940178753704, rtol=rtol, atol=atol)
    assert_allclose(dist.kurtosis, 2.1824677678598667, rtol=rtol, atol=atol)
    assert dist.name2 == "Competing risks using 2 distributions"
    assert_allclose(dist.quantile(0.2), 14.170859470541174, rtol=rtol, atol=atol)
    assert_allclose(dist.inverse_SF(q=0.7), 17.908811127053173, rtol=rtol, atol=atol)
    assert_allclose(dist.mean_residual_life(20), 9.862745898092886, rtol=rtol, atol=atol)
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
        [0.00210671, 0.00661657, 0.01947571, 0.02655321, 0.00474024, 0.00062978],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.CDF(xvals=xvals, show_plot=True),
        [0.0010001, 0.00999995, 0.09999943, 0.90000184, 0.99000021, 0.99900003],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.SF(xvals=xvals, show_plot=True),
        [0.9989999, 0.99000005, 0.90000057, 0.09999816, 0.00999979, 0.00099997],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.HF(xvals=xvals, show_plot=True),
        [0.00210882, 0.00668341, 0.02163966, 0.265537, 0.47403341, 0.62980068],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.CHF(xvals=xvals, show_plot=True),
        [1.00059934e-03, 1.00502826e-02, 1.05359884e-01, 2.30260350e00, 4.60519097e00, 6.90778668e00],
        rtol=rtol,
        atol=atol,
    )


def test_Mixture_Model():
    distributions = [Weibull_Distribution(alpha=30, beta=2), Normal_Distribution(mu=35, sigma=5)]
    dist = Mixture_Model(distributions=distributions, proportions=[0.6, 0.4])
    assert_allclose(dist.mean, 29.952084649328917, rtol=rtol, atol=atol)
    assert_allclose(dist.standard_deviation, 11.95293368817564, rtol=rtol, atol=atol)
    assert_allclose(dist.variance, 142.87262375392413, rtol=rtol, atol=atol)
    assert_allclose(dist.skewness, 0.015505959874527537, rtol=rtol, atol=atol)
    assert_allclose(dist.kurtosis, 3.4018343377801674, rtol=rtol, atol=atol)
    assert dist.name2 == "Mixture using 2 distributions"
    assert_allclose(dist.quantile(0.2), 19.085648329240094, rtol=rtol, atol=atol)
    assert_allclose(dist.inverse_SF(q=0.7), 24.540270766923847, rtol=rtol, atol=atol)
    assert_allclose(dist.mean_residual_life(20), 14.686456940211107, rtol=rtol, atol=atol)
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
        [0.0016309, 0.00509925, 0.01423464, 0.01646686, 0.00134902, 0.00016862],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.CDF(xvals=xvals, show_plot=True),
        [0.00099994, 0.00999996, 0.10000006, 0.90000056, 0.99000001, 0.999],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.SF(xvals=xvals, show_plot=True),
        [0.99900006, 0.99000004, 0.89999994, 0.09999944, 0.00999999, 0.001],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.HF(xvals=xvals, show_plot=True),
        [0.00163253, 0.00515076, 0.01581627, 0.16466956, 0.13490177, 0.16861429],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.CHF(xvals=xvals, show_plot=True),
        [1.00043950e-03, 1.00502998e-02, 1.05360581e-01, 2.30259070e00, 4.60517090e00, 6.90775056e00],
        rtol=rtol,
        atol=atol,
    )
