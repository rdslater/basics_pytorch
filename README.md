# basics_pytorch
This is my review of my fundamentals in pytorch.  I was using lightning for a long time and want to get back to basics so I can do GaNs and more of the hard problems.  Also like a challenge and review is good.

`bare_bones_regression.py`  - this is basic regression problem using the California Housing Dataset.  Basic Training loop so I can practice organizing my code
`compact_regression.py`  I worked with a coding agent to make a better structured file.  The bare bones was me "re-learning" raw pytorch so I had things scattered all over.  The other thing I like about this one is it switches on `torch.no_grad()` by passing an optimizer or not.  A little dangerous, but I like compact effcient code