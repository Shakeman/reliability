from numpy.testing import assert_allclose

from reliability.ALT_Fitters import Fit_Normal_Dual_Exponential, Fit_Weibull_Exponential, Fit_Weibull_Power
from reliability.Datasets import (
    MCF_2,
    ALT_load,
    ALT_load2,
    ALT_temperature,
    ALT_temperature2,
    ALT_temperature3,
    ALT_temperature4,
    ALT_temperature_humidity,
    ALT_temperature_voltage,
    ALT_temperature_voltage2,
    automotive,
    defective_sample,
    mileage,
    mixture,
    system_growth,
)
from reliability.Fitters import Fit_Weibull_2P, Fit_Weibull_DS, Fit_Weibull_Mixture
from reliability.Repairable_systems import MCF_nonparametric

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


def test_MCF_2():
    results = MCF_nonparametric(data=MCF_2().times)
    assert results.last_time == 3837


def test_alt_temp():
    results = Fit_Weibull_Exponential(
        failures=ALT_temperature().failures,
        failure_stress=ALT_temperature().failure_stresses,
        right_censored=ALT_temperature().right_censored,
        right_censored_stress=ALT_temperature().right_censored_stresses,
    )
    assert_allclose(results.a, 208.33402098466703, rtol=rtol, atol=atol)
    assert_allclose(results.b, 157.57367698539767, rtol=rtol, atol=atol)
    assert_allclose(results.beta, 1.399832090336735, rtol=rtol, atol=atol)
    assert_allclose(results.AICc, 689.3628062711795, rtol=rtol, atol=atol)


def test_alt_temp2():
    results = Fit_Weibull_Exponential(
        failures=ALT_temperature2().failures,
        failure_stress=ALT_temperature2().failure_stresses,
        right_censored=ALT_temperature2().right_censored,
        right_censored_stress=ALT_temperature2().right_censored_stresses,
    )
    assert_allclose(results.a, 589.6798006916221, rtol=rtol, atol=atol)
    assert_allclose(results.b, 24.30687828618073, rtol=rtol, atol=atol)
    assert_allclose(results.beta, 2.509716835190265, rtol=rtol, atol=atol)
    assert_allclose(results.AICc, 229.00733950073302, rtol=rtol, atol=atol)


def test_alt_temp3():
    results = Fit_Weibull_Exponential(
        failures=ALT_temperature3().failures, failure_stress=ALT_temperature3().failure_stresses
    )
    assert_allclose(results.a, 1862.6692701034315, rtol=rtol, atol=atol)
    assert_allclose(results.b, 58.83305869441037, rtol=rtol, atol=atol)
    assert_allclose(results.beta, 4.291582716275769, rtol=rtol, atol=atol)
    assert_allclose(results.AICc, 523.199373464674, rtol=rtol, atol=atol)


def test_alt_temp4():
    results = Fit_Weibull_Exponential(
        failures=ALT_temperature4().failures, failure_stress=ALT_temperature4().failure_stresses
    )
    assert_allclose(results.a, 5902.419825344584, rtol=rtol, atol=atol)
    assert_allclose(results.b, 0.00041397403586479676, rtol=rtol, atol=atol)
    assert_allclose(results.beta, 3.466098142009677, rtol=rtol, atol=atol)
    assert_allclose(results.AICc, 251.4973843590211, rtol=rtol, atol=atol)


def test_alt_load():
    results = Fit_Weibull_Power(failures=ALT_load().failures, failure_stress=ALT_load().failure_stresses)
    assert_allclose(results.a, 9401762.793282213, rtol=rtol, atol=atol)
    assert_allclose(results.beta, 2.522703529268249, rtol=rtol, atol=atol)
    assert_allclose(results.AICc, 264.63934122578297, rtol=rtol, atol=atol)


def test_alt_load2():
    results = Fit_Weibull_Power(
        failures=ALT_load2().failures,
        failure_stress=ALT_load2().failure_stresses,
        right_censored=ALT_load2().right_censored,
        right_censored_stress=ALT_load2().right_censored_stresses,
    )
    assert_allclose(results.a, 50176.77500484591, rtol=rtol, atol=atol)
    assert_allclose(results.beta, 2.886844300531308, rtol=rtol, atol=atol)
    assert_allclose(results.AICc, 164.5457691868064, rtol=rtol, atol=atol)


def test_alt_temp_volt():
    data = ALT_temperature_voltage()
    results = Fit_Normal_Dual_Exponential(
        failures=data.failures, failure_stress_1=data.failure_stress_temp, failure_stress_2=data.failure_stress_voltage
    )
    assert_allclose(results.a, 4345.647638568313, rtol=rtol, atol=atol)
    assert_allclose(results.b, 3.0514583174522096, rtol=rtol, atol=atol)
    assert_allclose(results.c, 0.0009352431954755059, rtol=rtol, atol=atol)
    assert_allclose(results.AICc, 155.26657670807614, rtol=rtol, atol=atol)


def test_alt_temp_volt2():
    data = ALT_temperature_voltage2()
    results = Fit_Normal_Dual_Exponential(
        failures=data.failures, failure_stress_1=data.failure_stress_temp, failure_stress_2=data.failure_stress_voltage
    )
    assert_allclose(results.a, 3736.1347199847964, rtol=rtol, atol=atol)
    assert_allclose(results.b, -28.351882768859106, rtol=rtol, atol=atol)
    assert_allclose(results.c, 0.6733124765303472, rtol=rtol, atol=atol)
    assert_allclose(results.AICc, 290.91988252498925, rtol=rtol, atol=atol)


def test_alt_temp_humidity():
    data = ALT_temperature_humidity()
    results = Fit_Normal_Dual_Exponential(
        failures=data.failures, failure_stress_1=data.failure_stress_temp, failure_stress_2=data.failure_stress_humidity
    )
    assert_allclose(results.a, 6398.279398789768, rtol=rtol, atol=atol)
    assert_allclose(results.b, 0.31744610580758836, rtol=rtol, atol=atol)
    assert_allclose(results.c, 6.834419843778612e-06, rtol=rtol, atol=atol)
    assert_allclose(results.AICc, 135.9958438480237, rtol=rtol, atol=atol)
