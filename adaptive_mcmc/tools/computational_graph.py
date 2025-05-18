from collections import deque


def count_graph_nodes(output_tensor):
    """
    Count the number of unique autograd Function nodes in the backward graph
    of `output_tensor`.
    """
    visited = set()
    queue = deque([output_tensor.grad_fn])
    count = 0

    while queue:
        fn = queue.popleft()
        if fn is None or fn in visited:
            continue
        visited.add(fn)
        count += 1
        # each fn.next_functions is a tuple of (next_fn, index)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)

    return count


def print_graph_with_tensors(output_tensor):
    """
    Traverse the autograd graph of `output_tensor` and print each unique
    Function node *and* any tensors it holds (saved_tensors or variable).
    """
    visited = set()
    queue = deque([output_tensor.grad_fn])

    while queue:
        fn = queue.popleft()
        if fn is None or fn in visited:
            continue
        visited.add(fn)

        # 1) Leaf‐gradient nodes have `.variable`
        if hasattr(fn, 'variable'):
            print(f"{type(fn).__name__} (id={id(fn)}):")
            print("   variable:", fn.variable.shape)

        # 2) Most other nodes expose `.saved_tensors`
        # saved = getattr(fn, 'saved_tensors', None)
        # if saved:
        #     for i, t in enumerate(saved):
        #         print(f"   saved_tensors[{i}]: {t.shape}")

        # enqueue parents
        for parent_fn, _ in fn.next_functions:
            if parent_fn is not None:
                queue.append(parent_fn)
