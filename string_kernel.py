import numpy as np

# Kernel defined by Lodhi et al. (2002)
def ssk(s, t, n, lbda, accum=False):
    dynamic = {}

    def k_prim(s, t, i):
        # print( "k_prim({},{},{})".format(s, t, i) )
        if i == 0:
            # print( "k_prim({},{},{}) => 1".format(s, t, i)  )
            return 1.
        if min(len(s), len(t)) < i:
            # print( "k_prim({},{},{}) => 0".format(s, t, i)  )
            return 0.
        if (s,t,i) in dynamic:
            return dynamic[(s,t,i)]

        x = s[-1]
        s_ = s[:-1]
        indices = [i for i, e in enumerate(t) if e == x]
        toret = lbda * k_prim(s_, t, i) \
              + sum( k_prim(s_, t[:j], i-1) * (lbda**(len(t)-j+1)) for j in indices )
        # print( "k_prim({},{},{}) => {}".format(s, t, i, toret) )
        dynamic[(s,t,i)] = toret
        return toret

    def k(s, t, n):
        # print( "k({},{},{})".format(s, t, n) )
        if n <= 0:
            raise "Error, n must be bigger than zero"
        if min(len(s), len(t)) < n:
            # print( "k({},{},{}) => 0".format(s, t, n) )
            return 0.
        x = s[-1]
        s_ = s[:-1]
        indices = [i for i, e in enumerate(t) if e == x]
        toret = k(s_, t, n) \
              + lbda**2 * sum( k_prim(s_, t[:j], n-1) for j in indices )
        # print( "k({},{},{}) => {}".format(s, t, n, toret) )
        return toret

    if accum:
        toret = sum( k(s, t, i) for i in range(1, min(n,len(s),len(t))+1) )
    else:
        toret = k(s, t, n)

    # print( len(dynamic) )
    return toret

def string_kernel(xs, ys, n, lbda):
    if len(xs.shape) != 2 or len(ys.shape) != 2 or xs.shape[1] != 1 or ys.shape[1] != 1:
        raise "The shape of the features is wrong, it must be (n,1)"

    lenxs, lenys = xs.shape[0], ys.shape[0]

    mat = np.zeros( (lenxs, lenys) )
    for i in range(lenxs):
        for j in range(lenys):
            mat[i,j] = ssk(xs[i,0], ys[j,0], n, lbda, accum=True)

    mat_xs = np.zeros( (lenxs, 1) )
    mat_ys = np.zeros( (lenys, 1) )

    for i in range(lenxs):
        mat_xs[i] = ssk(xs[i,0], xs[i,0], n, lbda, accum=True)
    for j in range(lenys):
        mat_ys[j] = ssk(ys[j,0], ys[j,0], n, lbda, accum=True)

    return np.divide(mat, np.sqrt(mat_ys.T * mat_xs))

if __name__ == '__main__':
    print("Testing...")
    lbda = .6
    assert abs( ssk("cat", "cart", 4, lbda, accum=True) - (3*lbda**2 + lbda**4 + lbda**5 + 2*lbda**7) ) < 1e-6
    assert ssk("science is organized knowledge", "wisdom is organized life", 4, 1, accum=True) == 20538.0

    xs = np.array( ["cat", "car", "cart", "camp", "shard"] ).reshape( (5,1) )
    ys = np.array( ["a", "cd"] ).reshape( (2,1) )
    assert string_kernel(xs, xs, 2, 1.)[0,0] == 1.

    print( string_kernel(xs, ys, 2, 1.) )
