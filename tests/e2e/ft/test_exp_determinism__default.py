# Temporary experiment entry (not part of the FT suite): see conftest_ft/experiment_determinism.py.

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.ft.conftest_ft.experiment_determinism import Variant, run_experiment

register_cuda_ci(
    est_time=3000,
    suite="stage-c-8-gpu-h200",
    labels=["ft-short"],
    hardware=["hopper", "blackwell"],
)

_VARIANT = Variant(
    name="default",
)

if __name__ == "__main__":
    run_experiment(_VARIANT)
