"""Run every Lab 2 step in order, then regenerate report.md."""
import print_versions
import sanity_checks
import train
import timing
import make_report

if __name__ == "__main__":
    for step in (print_versions, sanity_checks, train, timing, make_report):
        print(f"\n=== {step.__name__} ===")
        step.main()
