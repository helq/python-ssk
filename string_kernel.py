import numpy as np

# Kernel defined by Lodhi et al. (2002)
def ssk(s, t, n, lbda, accum=False):
    lens, lent = len(s), len(t)
    #k_prim = (-1)*np.ones( (n+1, lens, lent) )
    k_prim = np.zeros( (n, lens, lent) )
    indices = { x : [i for i, e in enumerate(t) if e == x] for x in set(s) }

    k_prim[0,:,:] = 1

    for i in range(1,n):
        for sj in range(i,lens):
            toret = 0.
            for tk in range(i,lent):
                if s[sj-1]==t[tk-1]: # trick taken from shogun implemantion of SSK
                    toret = lbda * (toret + lbda*k_prim[i-1,sj-1,tk-1])
                else:
                    toret *= lbda
                k_prim[i,sj,tk] = toret + lbda * k_prim[i, sj-1, tk]


    start = 0 if accum else n-1
    k = 0.
    for i in range(n):
        for sj in range(i,lens):
            for tk in range(i,lent):
                if s[sj]==t[tk]:
                    k += lbda*lbda*k_prim[i,sj,tk]

    # print( [len(list(i for (sj,tk,i) in k_prim if i==m-1)) for m in range(n)] )
    return k

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
