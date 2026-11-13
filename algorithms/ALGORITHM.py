import time

class Algorithm:
    """
    Algorithm        The parent class of algorithms.


    Algorithm properties:
        config      <object | None>     parameters of the algorithm
        problem     <BaseProblem>       the problem to solve
        starttime   <float>             clock start of each run
        result      <list[Move]>        move sequence of this run
        metric      <dict[str, float]>  relocations, crane_time, and runtime for each run
        
    Algorithm methods:
        __init__    store the parameters of this run set by the user
        solve       use the algorithm to solve a problem
        main        the main function of the algorithm
        parameter    read a named parameter of this run
    """
    

    def __init__(self, config=None):
        self.config = config

    def solve(self, problem):
        self.problem = problem
        self.starttime = time.perf_counter()
        self.result = self.main()
        
        
        relocations = sum(1 for step in self.result if step.destination is not None)
        self.metric = {
            "relocations": float(relocations),
            "crane_time": 0.0,
            "runtime": time.perf_counter() - self.starttime,
        }
        return self.result, self.metric
    
    def main(self):
        raise NotImplementedError

    def parameter(self, name, default, cast):
        given = {}
        if self.config is not None:
            given = getattr(self.config, "extra", None) or {}
        value = given.get(name, default)
        if value in (None, ""):
            value = default
        return cast(value)
