"""Powell's method"""

import math
import os
import threading
from contextlib import nullcontext
from queue import Queue
from typing import Callable, Tuple

import numpy
from tqdm.auto import tqdm, trange

from ..typing import NDArray

__R = -1 + math.sqrt(5) / 2
__c = 1 + __R


def __record_queue_vals(fpath: os.PathLike, queue: Queue):
    with open(fpath, "w") as f:
        while True:
            state, val = queue.get()
            if val is None:
                break
            vec = numpy.concatenate([state, val], axis=0)
            vec_str = ",".join([f"{val:.4}" for val in vec]) + "\n"
            f.write(vec_str)


def goldsect_bracket(
    f: Callable[[float], float], x1: float, incr: float, max_iter: int = 100
) -> Tuple[float, float]:
    f1 = f(x1)
    x2 = x1 + incr
    f2 = f(x2)
    # determine downhill direction, change sign if needed
    if f2 > f1:  # need to change direction of h
        incr = -incr
        x2 = x1 + incr
        f2 = f(x2)
        if f2 > f1:
            return x2, x1 - incr
    # search loop for bracketing
    for _ in range(max_iter):
        incr = __c * incr
        x3 = x2 + incr
        f3 = f(x3)
        if f3 > f2:
            return x1, x3
        x1 = x2
        x2 = x3
        f1 = f2
        f2 = f3
    raise Exception("golden section search bracketing did not find minimum")


def goldsect_search(
    f: Callable[[float], float], a: float, b: float, tol: float = 1e-9
) -> Tuple[float, float]:
    n_iter = int(math.ceil(-2.078087 * math.log(tol / abs(b - a))))
    # first telescoping
    x1 = __R * a + (1 - __R) * b
    x2 = (1 - __R) * a + __R * b
    f1, f2 = f(x1), f(x2)
    for _ in range(n_iter):
        if f1 > f2:
            a = x1
            x1, f1 = x2, f2
            x2 = (1 - __R) * a + __R * b
            f2 = f(x2)
        else:
            b = x2
            x2, f2 = x1, f1
            x1 = __R * a + (1 - __R) * b
            f1 = f(x1)
    return (x1, f1) if f1 < f2 else (x2, f2)


def powell_record(
    F: Callable[[NDArray], float],
    x: NDArray,
    rec_path: os.PathLike,
    max_iter: int = 30,
    search_incr: float = 0.1,
    tol: float = 1e-6,
    verbose: bool = False,
):
    # setup the writer thread
    rec_queue = Queue()
    rec_target = lambda: __record_queue_vals(rec_path, rec_queue)
    rec_thread = threading.Thread(name="writer", target=rec_target)
    rec_thread.start()
    # do the optimization, wrapping it in a try/except so that if we hit
    # `max_iter`, we still gracefully cleanup the writer thread
    try:
        state, n_iter = powell(F, x, max_iter, search_incr, tol, rec_queue, verbose)
    except Exception as e:
        print(e, flush=True)
        rec_thread.join()
        raise e
    rec_thread.join()
    return state, n_iter


def powell(
    F: Callable[[NDArray], float],
    x: NDArray,
    max_iter: int = 30,
    search_incr: float = 0.1,
    tol: float = 1e-6,
    rec_queue: Queue | None = None,
    verbose: bool = False,
) -> Tuple[NDArray, int]:
    if rec_queue is None:

        def _f1(v: NDArray, s: float) -> float:
            return F(x + s * v)
    else:

        def _f1(v: NDArray, s: float) -> float:
            state = x + s * v
            val = F(state)
            rec_queue.put((state, val))
            return val

    n_par = len(x)
    dec_f = numpy.zeros(n_par)
    u = numpy.eye(n_par)
    with tqdm(total=100) if verbose else nullcontext as pbar:
        for i in trange(max_iter, desc="cycle", leave=False):
            x_prev = x.copy()
            f_prev = F(x_prev)
            if verbose:
                pbar.n = int(math.floor((1 - f_prev) * 100))
                pbar.refresh()
            # for each parameter, define search direction and look
            # for decreases in F along that direction
            for j in trange(n_par, leave=False, desc="parloop"):
                v = u[j]
                f = lambda s: _f1(v, s)
                a, b = goldsect_bracket(f, 0.0, search_incr)
                s, f_min = goldsect_search(f, a, b)
                dec_f[j] = f_prev - f_min
                f_prev = f_min
                if verbose:
                    pbar.set_postfix({"par_idx": j})
                    pbar.n = int(math.floor((1 - f_prev) * 100))
                    pbar.refresh()
                x = x + s * v
            # last line search in this cycle
            v = x - x_prev  # v_{n+1} = x_0 - x_prev
            f = lambda s: _f1(v, s)
            a, b = goldsect_bracket(f, 0.0, search_incr)
            s, f_last = goldsect_search(f, a, b)
            if verbose:
                pbar.set_postfix_str("last")
                pbar.n = int(math.floor((1 - f_last) * 100))
                pbar.refresh()
            x = x + s * v
            # check for convergence, if converged, return
            if math.sqrt(numpy.dot(x - x_prev, x - x_prev) / max_iter) < tol:
                if rec_queue is not None:  # signal to kill recorder thread
                    rec_queue.put((None, None))
                return x, i + 1
            # if we got here, no convergence, need to update search directions
            i_max = numpy.argmax(dec_f)  # index for biggest decrease
            for j in range(i_max, n_par - 1):
                u[j] = u[j + 1]
            u[n_par - 1] = v
        if rec_queue is not None:  # signal to kill recorder thread
            rec_queue.put((None, None))
        raise Exception("powell's method did not converge")
