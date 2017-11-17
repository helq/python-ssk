# Fast String Kernel (SSK) implementation for python #

Implementation of string kernel as described on [Lodhi et al (2002)][paper] (aka. SSK).

[paper]: http://jmlr.csail.mit.edu/papers/volume2/lodhi02a/lodhi02a.pdf

The main function is written cython (python with some C type-level annotations) which
makes it almost as fast as the fastest implementation I found for SSK. The fastest
SSK implementation I found is shogun's [SubsequenceStringKernel.cpp][shogunimp], in fact,
I copied some small trick they did in their implementation of SSK for this implementation.

[shogunimp]: https://github.com/shogun-toolbox/shogun/blob/b1cf826876093c3b26346116c28bd077e4db6b0c/src/shogun/kernel/string/SubsequenceStringKernel.cpp#L87

## Requisites ##

- Python
- Cython
- Numpy

## How to use ##

Take a look at `main.py`, that should suffice.

A simple wrapper around `string_kernel()` to use this kernel in, for example, scikit-learn
is easy, just:

```python
lambda_decay = .8
max_substring = 5
my_ssk_kernel = lambda l, r: string_kernel(l, r, max_substring, lambda_decay)
```

## TODO ##

* `string_kernel` should accept arbitrary lists not only python strings

## How does actually ssk() works? ##

If you're interested on how the recursive functions from the [paper][] got to converted
into the very confusing, three-looped, written in C-ython function in this repository, I
recommend you to read the code-changes through the first to fourth commits, happy reading
;)

## License ##

I wanted to use [WTFPL](http://www.wtfpl.net/) to license the code, but CC0 is recommended
over it as CC0 is [more][fsfwtfpl] ["legal"][fsfunlicense] ;)

[fsfwtfpl]: https://www.gnu.org/licenses/license-list.html#WTFPL
[fsfunlicense]: https://www.gnu.org/licenses/license-list.html#Unlicense
