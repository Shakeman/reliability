import numpy as np
from numpy.testing import assert_allclose

from reliability.Distributions import (
    Beta_Distribution,
)

atol = 1e-8
rtol = 1e-7


def test_Beta_Distribution():
    dist = Beta_Distribution(alpha=5, beta=2)
    dist.plot()
    dist.stats()
    assert_allclose(dist.mean, 0.7142857142857143, rtol=rtol, atol=atol)
    assert_allclose(dist.standard_deviation, 0.15971914124998499, rtol=rtol, atol=atol)
    assert_allclose(dist.variance, 0.025510204081632654, rtol=rtol, atol=atol)
    assert_allclose(dist.skewness, -0.5962847939999439, rtol=rtol, atol=atol)
    assert_allclose(dist.kurtosis, 2.88, rtol=rtol, atol=atol)
    assert dist.param_title_long == "Beta Distribution (α=5.0,β=2.0)"
    assert_allclose(dist.quantile(0.2), 0.577552475153728, rtol=rtol, atol=atol)
    assert_allclose(dist.inverse_SF(q=0.7), 0.6396423096199797, rtol=rtol, atol=atol)
    assert_allclose(dist.mean_residual_life(0.5), 0.2518796992481146, rtol=rtol, atol=atol)
    xvals = np.array(
        [
            0,
            dist.quantile(0.001),
            dist.quantile(0.01),
            dist.quantile(0.1),
            dist.quantile(0.9),
            dist.quantile(0.99),
            dist.quantile(0.999),
        ],
    )
    assert_allclose(
        dist.PDF(xvals=xvals, show_plot=True),
        [
            0.0,
            0.026583776746547504,
            0.15884542294682907,
            0.8802849346924463,
            1.883276908534153,
            0.7203329063913153,
            0.23958712288762668,
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
            0.026610387133681187,
            0.16044992216851423,
            0.9780943718804959,
            18.832769085341553,
            72.03329063913147,
            239.58712288762646,
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
            2.3025850929940472,
            4.605170185988091,
            6.907755278982136,
        ],
        rtol=rtol,
        atol=atol,
    )
