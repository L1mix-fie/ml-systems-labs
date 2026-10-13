"""Run every Lab 1 step in order, then regenerate report.md."""
import print_versions
import train_baselines
import measure_cost
import budget_fit
import make_report

if __name__ == "__main__":
    for step in (print_versions, train_baselines, measure_cost, budget_fit, make_report):
        print(f"\n=== {step.__name__} ===")
        step.main()
