from numpy.testing import assert_allclose

from reliability.Datasets import automotive, defective_sample, mileage, mixture, system_growth
from reliability.Fitters import Fit_Weibull_2P, Fit_Weibull_DS, Fit_Weibull_Mixture

atol = 1e-3
rtol = 1e-3


def test_automotive():
    results = Fit_Weibull_2P(failures=automotive().failures, right_censored=automotive().right_censored)
    assert_allclose(results.alpha, 134242.81713449836, rtol=rtol, atol=atol)
    assert_allclose(results.beta, 1.1558579098938948, rtol=rtol, atol=atol)
    assert_allclose(results.gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(results.AICc, 262.3763276243238, rtol=rtol, atol=atol)
    assert_allclose(results.BIC, 264.8157306047226, rtol=rtol, atol=atol)
    assert_allclose(results.loglik, -128.97387809787617, rtol=rtol, atol=atol)
    assert_allclose(results.AD, 35.6074776540121, rtol=rtol, atol=atol)
    assert_allclose(results.Cov_alpha_beta, -6296.482770776141, rtol=rtol, atol=atol)


def test_mileage():
    results = Fit_Weibull_2P(failures=mileage().failures)
    assert_allclose(results.alpha, 33518.72711380954, rtol=rtol, atol=atol)
    assert_allclose(results.beta, 3.1346123913787327, rtol=rtol, atol=atol)
    assert_allclose(results.gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(results.AICc, 2136.5291166520465, rtol=rtol, atol=atol)


def test_system_growth():
    results = Fit_Weibull_2P(failures=system_growth().failures)
    assert_allclose(results.alpha, 223.52129873129442, rtol=rtol, atol=atol)
    assert_allclose(results.beta, 1.0418121869777046, rtol=rtol, atol=atol)
    assert_allclose(results.gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(results.AICc, 285.96592563737926, rtol=rtol, atol=atol)


def test_defective_sample():
    results = Fit_Weibull_DS(failures=defective_sample().failures, right_censored=defective_sample().right_censored)
    assert_allclose(results.alpha, 170.9830814106804, rtol=rtol, atol=atol)
    assert_allclose(results.beta, 1.301084338811416, rtol=rtol, atol=atol)
    assert_allclose(results.DS, 0.1248205589053823, rtol=rtol, atol=atol)
    assert_allclose(results.AICc, 23961.321843508023, rtol=rtol, atol=atol)


def test_mixture():
    results = Fit_Weibull_Mixture(failures=mixture().failures, right_censored=mixture().right_censored)
    assert_allclose(results.alpha_1, 177414.58191845534, rtol=rtol, atol=atol)
    assert_allclose(results.beta_1, 1.1928101148005372, rtol=rtol, atol=atol)
    assert_allclose(results.alpha_2, 68578.46269607474, rtol=rtol, atol=atol)
    assert_allclose(results.beta_2, 3.035100882926186, rtol=rtol, atol=atol)
