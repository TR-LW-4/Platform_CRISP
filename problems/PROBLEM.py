class BaseProblem:
    """
    BaseProblem        The parent class of problems.


    BaseProblem properties:
        num_bays        <int>   number of bays
        num_rows        <int>   number of rows
        max_tiers       <int>   maximum tier of a stack
        num_containers  <int>   number of containers
        stacks          <list>  initial stacks, bottom to top
        retrieval_order  <list>  retrieval order of this layout
       
       
    BaseProblem methods:
        __init__    store the parameters of this run set by the user
       
    """

    def __init__(
        self,
        num_bays=1,
        num_rows=1,
        max_tiers=1,
        num_containers=0,
        stacks=None,
        retrieval_order=None,
    ):
        self.num_bays = num_bays
        self.num_rows = num_rows
        self.max_tiers = max_tiers
        self.num_containers = num_containers
        self.stacks = [] if stacks is None else stacks
        self.retrieval_order = [] if retrieval_order is None else retrieval_order
        


