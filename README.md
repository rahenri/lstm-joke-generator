# LSTM joke generator trained from reddit jokes

To run it, fetch the dataset from the repo:
<https://github.com/taivop/joke-dataset> and put the reddit jokes file in the
same directory as the main.py script, run it and enjoy. Training could take
long time, gpu support for tensorflow is highly recommended.

Dependencies:
* Tensorflow
* Keras

See some generated jokes at [generated.txt](../master/generated.txt). They are
non sense, but one can see that it learned to generate english words without
many grammar mistakes. Maybe a better model or a larger dataset would help with
that.
