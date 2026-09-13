"""Run schedule candidates through the common rounded, true-type harness."""
import bench
from schedule_candidate import CandidateState
from online import State
bench.build=lambda method:State() if method=='v4' else CandidateState(strategy=method)
if __name__=='__main__':bench.main()
