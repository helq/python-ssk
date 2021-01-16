# How to run (linux):
# > python main.py
#
# If Cython can't find the NumPy libraries at compilation time:
# > CFLAGS=-I/usr/lib/python3.9/site-packages/numpy/core/include python main

import pyximport; pyximport.install()  # noqa: E702

from string_kernel import ssk, string_kernel
import numpy as np


if __name__ == '__main__':
    print("Testing...")
    lbda = .6
    # We can compute this kernel by hand and the result should be the thing on the
    # second column
    assert abs(ssk("cat", "cart", 4, lbda, accum=True)
               - (3*lbda**2 + lbda**4 + lbda**5 + 2*lbda**7)) < 1e-6
    # From paper
    assert ssk("science is organized knowledge",
               "wisdom is organized life", 4, 1, accum=True) == 20538.0

    xs = np.array(["cat", "car", "cart", "camp", "shard"]).reshape((5, 1))
    ys = np.array(["a", "cd"]).reshape((2, 1))
    print(string_kernel(xs, ys, 2, 1.))
    assert abs(string_kernel(xs, ys, 2, 1.)[0, 0] - 0.40824829) < 1e-6

    test = "This is a very long string, just to test how fast this implementation " \
        "of ssk is. It should look like the computation tooks no time, unless you're" \
        " running this in a potato pc"
    print(ssk(test, test, 30, .8, accum=True))
