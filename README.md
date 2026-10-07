# basics_pytorch
This is my review of my fundamentals in pytorch.  I was using lightning for a long time and want to get back to basics so I can do GaNs and more of the hard problems.  Also like a challenge and review is good.

`bare_bones_regression.py`  - this is basic regression problem using the California Housing Dataset.  Basic Training loop so I can practice organizing my code.  This will probably become my introductory teaching example, so lots of code comments in here

`compact_regression.py`  I worked with a coding agent to make a better structured file.  The bare bones was me "re-learning" raw pytorch so I had things scattered all over.  The other thing I like about this one is it switches on `torch.no_grad()` by passing an optimizer or not.  A little dangerous, but I like compact effcient code.  This is a refactor of the bare_bones script--does the same thing, just better readibility.

I added in early stopping on 10/7--yanking it out and adding early stopping separate

`compact_classification` I just used the digits set for a basic classification.  here I made the fit function more explicit so its very visible what is going on.  Just a classification example

`EarlyStopping.py` this contains a claude generated Early stopping.  Yeah I cheated and was lazy, but it made a very nice feature rich early stopping module.  Don't need to re-invent the wheel here.

`compact_regression_earlystopping.py` adds in Early Stopping to compact_regression.py
## Next Steps
- ~~Early Stopping~~
- Model Checkpoints (with Lightning style naming)
- Tensorboard and CSV logging (Printing is nice, logging is better)
- TQDM profress Bars. (I'm fancy!)

## Lineage of Files
Yes I could just store this as history/commits, but this is for education so its good to compare how the scripts "grow" as specific features get added in

`bare_bones_regression.py` -> (Refactor) -> `compact_regression.py` -> (Add Early Stopping) -> `compact_regression_earlystopping.py`

`compact_regression.py` (Explicit Val Call + Change to Classification) ->  `compact_classification.py`