from reliability.Repairable_systems import (
    MCF_nonparametric,
)

data = [1, 3, 5]
results = MCF_nonparametric(data=data)

results.print_results()
