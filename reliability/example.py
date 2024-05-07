import matplotlib.pyplot as plt

from reliability.ALT_Fitters import Fit_Weibull_Power
from reliability.Datasets import ALT_load2

Fit_Weibull_Power(failures=ALT_load2().failures, failure_stress=ALT_load2().failure_stresses, right_censored=ALT_load2().right_censored, right_censored_stress=ALT_load2().right_censored_stresses, use_level_stress=60)
plt.show()
