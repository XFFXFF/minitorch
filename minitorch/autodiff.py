from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Dict

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals = list(vals)
    f_x = f(*vals) 
    vals[arg] = vals[arg] + epsilon
    f_x_plus_epsilon = f(*vals)
    return (f_x_plus_epsilon - f_x) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    topo_order: List[Variable] = []

    def insert_variable(v: Variable, topo_order: List[Variable]):
        topo_order_ids = [v.unique_id for v in topo_order]
        for p in v.parents:
            if p.unique_id not in topo_order_ids:
                insert_variable(p, topo_order)
        topo_order.append(v)
    insert_variable(variable, topo_order)
    return topo_order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    node_to_deriv: Dict[int, Any] = {variable.unique_id: [deriv]}
    for v in reversed(topological_sort(variable)):
        if v.is_leaf():
            continue
        v_bar = sum(node_to_deriv[v.unique_id])
        for p, d in v.chain_rule(v_bar):
            node_to_deriv[p.unique_id] = node_to_deriv.get(p.unique_id, []) + [d]
            if p.is_leaf():
                p.accumulate_derivative(d)


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
