# A ptxas-compiled kernel runs end to end and checked on every part, built for each part's target;
# the larger run reports the simulator's speed.
lake build axpy && lake exe axpy --gpu all && lake exe axpy 65536
