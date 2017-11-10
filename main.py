import pyximport; pyximport.install()
from string_kernel import ssk, string_kernel

import numpy as np

if __name__ == '__main__':
    print("Testing...")
    lbda = .6
    assert abs( ssk("cat", "cart", 4, lbda, accum=True) - (3*lbda**2 + lbda**4 + lbda**5 + 2*lbda**7) ) < 1e-6
    assert ssk("science is organized knowledge", "wisdom is organized life", 4, 1, accum=True) == 20538.0

    xs = np.array( ["cat", "car", "cart", "camp", "shard"] ).reshape( (5,1) )
    ys = np.array( ["a", "cd"] ).reshape( (2,1) )
    print( string_kernel(xs, ys, 2, 1.) )
    assert abs( string_kernel(xs, ys, 2, 1.)[0,0] - 0.40824829 ) < 1e-6

    test = "the decision means a ruling could be made nearly two months before the regular season begins, time for the sides to work out a deal without delaying the season."
    print( ssk(test, test, 30, .8) )
