# Fast String Kernel (SSK) implementation for python #

Implementation of string kernel as described in [Lodhi et al (2002)][paper] (aka. SSK).

[paper]: http://jmlr.csail.mit.edu/papers/volume2/lodhi02a/lodhi02a.pdf

The main function is written Cython (Python with C type-level annotations) and is the
second fastest implementation of the SSK Kernel that I know of. The fastest
SSK implementation I found is Shogun's [SubsequenceStringKernel.cpp][shogunimp].
I copied a small trick from Shogun's implementation that speeds the computation
substantially.

[shogunimp]: https://github.com/shogun-toolbox/shogun/blob/b1cf826876093c3b26346116c28bd077e4db6b0c/src/shogun/kernel/string/SubsequenceStringKernel.cpp#L87

## Requisites ##

- Python
- Cython
- Numpy

## How to use ##

Take a look at `main.py`. That hopefully should suffice.

As an example, the following is a simple wrapper around `string_kernel()` to use SSK in scikit-learn:

```python
def get_ssk_kernel_for_scikit(max_substring, lambda_decay):
    def strker(il, ir):
        #print("Shape of gramm matrix to create ({},{})".format(len(il), len(ir)))
        # assuming that `il` and `ir` are lists of strings.
        # `len(il)` may fail to give you the size real size if you're using `np.array`s
        # the idea is to reshape your data to be `np.array`s of sizes (n, 1) and (m, 1)
        l = np.array(il).reshape((len(il), 1))
        r = np.array(ir).reshape((len(ir), 1))
        return string_kernel(l, r, max_substring, lambda_decay)
    return strker

max_substring = 5
lambda_decay = .8

my_ssk_kernel = get_ssk_kernel_for_scikit(max_substring, lambda_decay)
```

## TODO ##

* `string_kernel` should accept arbitrary lists not only python strings and numpy arrays

## How does `ssk()` actually works? ##

If you're interested on how the recursive functions from the [paper][] got converted into
the very confusing, three-looped Cython function, I recommend you to check the changes in
the code from the [first][] to the [fourth][] commits.

[first]: https://github.com/helq/python-ssk/commit/6acee597ff37f7e7e12dd8651421a4d34c5dad70
[fourth]: https://github.com/helq/python-ssk/commit/28a3bd1db2899ac35e3db630aa66c92ec081591e

## License ##

I wanted to use [WTFPL](http://www.wtfpl.net/) to license the code, but CC0 is recommended
over it as CC0 is [more][fsfwtfpl] ["legal"][fsfunlicense] ;)

[fsfwtfpl]: https://www.gnu.org/licenses/license-list.html#WTFPL
[fsfunlicense]: https://www.gnu.org/licenses/license-list.html#Unlicense
