import numpy as np
import pytest

from reliability.Distributions import Beta_Distribution, Exponential_Distribution, Gumbel_Distribution
from reliability.Utils._input_checking_utils import distributions_input_checking, fitters_input_checking


def test_fitters_input_checking_init():
    dist = "Weibull_2P"
    failures = np.array([1, 2, 3, 4, 5])
    method = "MLE"
    right_censored = np.array([6, 7, 8])
    optimizer = "TNC"
    CI = 0.95
    quantiles = [0.1, 0.5, 0.9]
    force_beta = 2.0
    force_sigma = 1.0
    CI_type = "reliability"

    fitters_input = fitters_input_checking(
        dist=dist,
        failures=failures,
        method=method,
        right_censored=right_censored,
        optimizer=optimizer,
        CI=CI,
        quantiles=quantiles,
        force_beta=force_beta,
        force_sigma=force_sigma,
        CI_type=CI_type,
    )

    assert np.array_equal(fitters_input.failures, failures)
    assert np.array_equal(fitters_input.right_censored, right_censored)
    assert fitters_input.CI == CI
    assert fitters_input.method == method
    assert fitters_input.optimizer == optimizer
    assert np.array_equal(fitters_input.quantiles, quantiles)
    assert fitters_input.force_beta == force_beta
    assert fitters_input.force_sigma == force_sigma
    assert fitters_input.CI_type == CI_type


def test_distributions_input_checking():
    # Test Case 1 all inputs provided, CI_X and CI_Y cannot be provided same time
    dist = Gumbel_Distribution(mu=50, sigma=8)
    xvals = np.array([1, 2, 3, 4, 5])
    xmin = 0
    xmax = 10
    show_plot = True
    plot_CI = True
    CI_type = "time"
    CI = 0.9
    CI_y = [0.1, 0.2, 0.3]
    CI_x = [1.0, 2.0, 3.0]

    input_checking = distributions_input_checking(
        dist=dist,
        func="PDF",
        xvals=xvals,
        xmin=xmin,
        xmax=xmax,
        show_plot=show_plot,
        plot_CI=plot_CI,
        CI_type=CI_type,
        CI=CI,
        CI_y=CI_y,
        CI_x=CI_x,
    )

    assert (input_checking.xvals == xvals).all()
    assert input_checking.xmin == xmin
    assert input_checking.xmax == xmax
    assert input_checking.show_plot == show_plot
    assert input_checking.plot_CI == plot_CI
    assert input_checking.CI_type == CI_type
    assert input_checking.CI == CI
    assert (input_checking.CI_y == CI_y).all()
    assert input_checking.CI_x is None  # CI_X defaults to None if CI_Y provide

    # Test Case 2 CI_X provided only
    input_checking = distributions_input_checking(
        dist=dist,
        func="PDF",
        xvals=xvals,
        xmin=xmin,
        xmax=xmax,
        show_plot=show_plot,
        plot_CI=plot_CI,
        CI_type=CI_type,
        CI=CI,
        CI_x=CI_x,
    )

    assert (input_checking.xvals == xvals).all()
    assert input_checking.xmin == xmin
    assert input_checking.xmax == xmax
    assert input_checking.show_plot == show_plot
    assert input_checking.plot_CI == plot_CI
    assert input_checking.CI_type == CI_type
    assert input_checking.CI == CI
    assert (input_checking.CI_x == CI_x).all()

    # Test Case 3 invalid func provided
    with pytest.raises(ValueError):
        input_checking = distributions_input_checking(
            dist=dist,
            func="abc",
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
            show_plot=show_plot,
            plot_CI=plot_CI,
            CI_type=CI_type,
            CI=CI,
            CI_x=CI_x,
        )
    # Test Case 4 invalid xvals provided
    with pytest.raises(ValueError):
        input_checking = distributions_input_checking(
            dist=dist,
            func="PDF",
            xvals="abc",  # type: ignore
            xmin=xmin,
            xmax=xmax,
            show_plot=show_plot,
            plot_CI=plot_CI,
            CI_type=CI_type,
            CI=CI,
            CI_x=CI_x,
        )

    # Test Case 5 invalid xmin provided
    with pytest.raises(ValueError):
        input_checking = distributions_input_checking(
            dist=dist,
            func="PDF",
            xvals=xvals,
            xmin="abc",  # type: ignore
            xmax=xmax,
            show_plot=show_plot,
            plot_CI=plot_CI,
            CI_type=CI_type,
            CI=CI,
            CI_x=CI_x,
        )

    # Test Case 6 invalid xmax provided
    with pytest.raises(ValueError):
        input_checking = distributions_input_checking(
            dist=dist,
            func="PDF",
            xvals=xvals,
            xmin=xmin,
            xmax="abc",  # type: ignore
            show_plot=show_plot,
            plot_CI=plot_CI,
            CI_type=CI_type,
            CI=CI,
            CI_x=CI_x,
        )

    # Test Case 7 invalid CI provided
    with pytest.raises(ValueError):
        input_checking = distributions_input_checking(
            dist=dist,
            func="PDF",
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
            show_plot=show_plot,
            plot_CI=plot_CI,
            CI_type=CI_type,
            CI="abc",  # type: ignore
            CI_x=CI_x,
        )

    # Test Case 8 invalid CI_y provided
    with pytest.raises(ValueError):
        input_checking = distributions_input_checking(
            dist=dist,
            func="PDF",
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
            show_plot=show_plot,
            plot_CI=plot_CI,
            CI_type=CI_type,
            CI=CI,
            CI_y="abc",  # type: ignore
        )

    # Test Case 9 invalid CI_x provided
    with pytest.raises(ValueError):
        input_checking = distributions_input_checking(
            dist=dist,
            func="PDF",
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
            show_plot=show_plot,
            plot_CI=plot_CI,
            CI_type=CI_type,
            CI=CI,
            CI_x="abc",  # type: ignore
        )

    # Test Case 10 invalid CI_type provided
    with pytest.raises(ValueError):
        input_checking = distributions_input_checking(
            dist=dist,
            func="PDF",
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
            show_plot=show_plot,
            plot_CI=plot_CI,
            CI_type=123,  # type: ignore
            CI=CI,
            CI_x=CI_x,
        )

    # Test Case 11 invalid show_plot provided
    with pytest.raises(ValueError):
        input_checking = distributions_input_checking(
            dist=dist,
            func="PDF",
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
            show_plot="abc",  # type: ignore
            plot_CI=plot_CI,
            CI_type=CI_type,
            CI=CI,
            CI_x=CI_x,
        )

    # Test Case 12 invalid plot_CI provided
    with pytest.raises(ValueError):
        input_checking = distributions_input_checking(
            dist=dist,
            func="PDF",
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
            show_plot=show_plot,
            plot_CI="abc",  # type: ignore
            CI_type=CI_type,
            CI=CI,
            CI_x=CI_x,
        )

    # Test Case 13 CI is true
    input_checking = distributions_input_checking(
        dist=dist,  # type: ignore
        func="PDF",
        xvals=xvals,
        xmin=xmin,
        xmax=xmax,
        show_plot=show_plot,
        plot_CI=plot_CI,
        CI_type=CI_type,
        CI=True,
        CI_x=CI_x,
    )
    assert input_checking.CI == 0.95

    # Test Case 14 CI is false
    input_checking = distributions_input_checking(
        dist=dist,  # type: ignore
        func="PDF",
        xvals=xvals,
        xmin=xmin,
        xmax=xmax,
        show_plot=show_plot,
        plot_CI=plot_CI,
        CI_type=CI_type,
        CI=False,
        CI_x=CI_x,
    )
    assert input_checking.CI == 0.95
    assert input_checking.plot_CI is False

    # Test Case 15 CI is None
    input_checking = distributions_input_checking(
        dist=dist,  # type: ignore
        func="PDF",
        xvals=xvals,
        xmin=xmin,
        xmax=xmax,
        show_plot=show_plot,
        plot_CI=plot_CI,
        CI_type=CI_type,
        CI=None,
        CI_x=CI_x,
    )
    assert input_checking.CI == 0.95

    # Test Case 16 CI is above 1
    with pytest.raises(ValueError):
        input_checking = distributions_input_checking(
            dist=dist,
            func="PDF",
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
            show_plot=show_plot,
            plot_CI=plot_CI,
            CI_type=CI_type,
            CI=1.1,
            CI_x=CI_x,
        )

    # Test Case 17 xmin is None, xmax is None, and xvals is float
    input_checking = distributions_input_checking(
        dist=dist,  # type: ignore
        func="PDF",
        xvals=2.0,
        xmin=None,
        xmax=None,
        show_plot=show_plot,
        plot_CI=plot_CI,
        CI_type=CI_type,
        CI=None,
        CI_x=CI_x,
    )
    assert input_checking.X == 2.0
    assert input_checking.show_plot is False

    # Test Case 18 all inputs provided, CI_X and CI_Y cannot be provided same time and CI_Type is reliability
    input_checking = distributions_input_checking(
        dist=dist,
        func="PDF",
        xvals=xvals,
        xmin=xmin,
        xmax=xmax,
        show_plot=show_plot,
        plot_CI=plot_CI,
        CI_type="reliability",
        CI=CI,
        CI_y=CI_y,
        CI_x=CI_x,
    )
    assert input_checking.CI_y is None

    # Test Case 19 all inputs provided, exponential distribution used, invalid CI_Type given, and dist.Z provided, while CI is None
    input_checking = distributions_input_checking(
        dist=Exponential_Distribution(Lambda=0.2, gamma=10, CI=0.95),
        func="PDF",
        xvals=xvals,
        xmin=xmin,
        xmax=xmax,
        show_plot=None,
        plot_CI=plot_CI,
        CI_type="abc",  # type: ignore
        CI=None,
        CI_y=CI_y,
        CI_x=CI_x,
    )
    assert input_checking.CI_type is None
    assert input_checking.show_plot is True

    # Test Case 20 all inputs provided, beta distribution used,  invalid CI_Type given
    input_checking = distributions_input_checking(
        dist=Beta_Distribution(alpha=5, beta=2),
        func="PDF",
        xvals=None,
        xmin=xmin,
        xmax=0.9,
        show_plot=show_plot,
        plot_CI=plot_CI,
        CI_type="abc",  # type: ignore
        CI=None,
        CI_y=CI_y,
        CI_x=CI_x,
    )
    assert input_checking.CI_type is None

    # Test Case 21 CI_Type None
    input_checking = distributions_input_checking(
        dist=dist,
        func="PDF",
        xvals=xvals,
        xmin=xmin,
        xmax=xmax,
        show_plot=show_plot,
        plot_CI=plot_CI,
        CI_type=None,
        CI=CI,
        CI_y=CI_y,
        CI_x=CI_x,
    )
    assert input_checking.CI_type == "time"

    # Test Case 22 CI_Type is in No_CI_Array
    input_checking = distributions_input_checking(
        dist=dist,
        func="PDF",
        xvals=xvals,
        xmin=xmin,
        xmax=xmax,
        show_plot=show_plot,
        plot_CI=plot_CI,
        CI_type="none",  # type: ignore
        CI=CI,
        CI_y=CI_y,
        CI_x=CI_x,
    )
    assert input_checking.CI_type is None

    # Test Case 23 CI_Type is invalid string
    input_checking = distributions_input_checking(
        dist=dist,
        func="PDF",
        xvals=xvals,
        xmin=xmin,
        xmax=xmax,
        show_plot=show_plot,
        plot_CI=plot_CI,
        CI_type="abc",  # type: ignore
        CI=CI,
        CI_y=CI_y,
        CI_x=CI_x,
    )
    assert input_checking.CI_type is None

    # Test Case 24 CI_x checking
    with pytest.raises(ValueError):
        input_checking = distributions_input_checking(
            dist=Exponential_Distribution(Lambda=0.2, gamma=10),
            func="PDF",
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
            show_plot=show_plot,
            plot_CI=plot_CI,
            CI_type=CI_type,
            CI=CI,
            CI_y=None,
            CI_x=-0.9,
        )

    input_checking = distributions_input_checking(
        dist=dist,
        func="PDF",
        xvals=xvals,
        xmin=xmin,
        xmax=xmax,
        show_plot=show_plot,
        plot_CI=plot_CI,
        CI_type=CI_type,
        CI=CI,
        CI_y=None,
        CI_x=0.9,
    )
    assert input_checking.CI_x == np.array(np.float64(0.9))

    input_checking = distributions_input_checking(
        dist=dist,
        func="PDF",
        xvals=xvals,
        xmin=xmin,
        xmax=xmax,
        show_plot=show_plot,
        plot_CI=plot_CI,
        CI_type=CI_type,
        CI=CI,
        CI_y=None,
        CI_x=CI_x,
    )
    assert (input_checking.CI_x == np.array(CI_x)).all()

    with pytest.raises(ValueError):
        input_checking = distributions_input_checking(
            dist=Exponential_Distribution(Lambda=0.2, gamma=10),
            func="PDF",
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
            show_plot=show_plot,
            plot_CI=plot_CI,
            CI_type=CI_type,
            CI=CI,
            CI_y=None,
            CI_x=[-0.9, 0.2],
        )

    # Test Case 24 CI_y checking
    with pytest.raises(ValueError):
        input_checking = distributions_input_checking(
            dist=dist,
            func="PDF",
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
            show_plot=show_plot,
            plot_CI=plot_CI,
            CI_type=CI_type,
            CI=CI,
            CI_y=-0.9,
            CI_x=None,
        )
    with pytest.raises(ValueError):
        input_checking = distributions_input_checking(
            dist=dist,
            func="CDF",
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
            show_plot=show_plot,
            plot_CI=plot_CI,
            CI_type=CI_type,
            CI=CI,
            CI_y=1.2,
            CI_x=None,
        )

    input_checking = distributions_input_checking(
        dist=dist,
        func="PDF",
        xvals=xvals,
        xmin=xmin,
        xmax=xmax,
        show_plot=show_plot,
        plot_CI=plot_CI,
        CI_type=CI_type,
        CI=CI,
        CI_y=0.9,
        CI_x=None,
    )
    assert input_checking.CI_y == np.array(np.float64(0.9))

    input_checking = distributions_input_checking(
        dist=dist,
        func="PDF",
        xvals=xvals,
        xmin=xmin,
        xmax=xmax,
        show_plot=show_plot,
        plot_CI=plot_CI,
        CI_type=CI_type,
        CI=CI,
        CI_y=CI_y,
        CI_x=None,
    )
    assert (input_checking.CI_y == np.array(CI_y)).all()

    with pytest.raises(ValueError):
        input_checking = distributions_input_checking(
            dist=dist,
            func="PDF",
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
            show_plot=show_plot,
            plot_CI=plot_CI,
            CI_type=CI_type,
            CI=CI,
            CI_y=[-0.9, 0.2],
            CI_x=None,
        )

    with pytest.raises(ValueError):
        input_checking = distributions_input_checking(
            dist=dist,
            func="CDF",
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
            show_plot=show_plot,
            plot_CI=plot_CI,
            CI_type=CI_type,
            CI=CI,
            CI_y=[0.9, 1.2],
            CI_x=None,
        )
