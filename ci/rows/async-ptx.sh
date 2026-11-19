# The same async kernels' PTX through the interpreter, against the same oracles; the interpreter's
# random schedules with held-back asynchronous completions (sched-tests).
lake build async-ptx sched-tests && lake exe async-ptx && lake exe sched-tests
