"""Same isolated CPU check, with legacy eligibility filtering before draws."""
import positive_sampler_v28 as base
import positive_sampler_v28a as eligible
import smoke_connected_learning_cpu_v28 as smoke

if __name__=='__main__':
    base.initialize=eligible.initialize
    smoke.main()
