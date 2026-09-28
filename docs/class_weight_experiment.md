# Class weighting experiment

Output of `experiments/compare_class_weights.py`, run once on the 16-class setup (× and x as separate classes), before the thinning fix in `src/preprocessing.py`. All numbers are on the validation set.

```
none      val_acc=0.9717  ×: P=0.000 R=0.000  x: P=0.863 R=0.987  ÷: R=0.870  other14_acc=0.9892
sqrt      val_acc=0.9711  ×: P=0.500 R=0.278  x: P=0.891 R=0.952  ÷: R=0.870  other14_acc=0.9877
balanced  val_acc=0.9503  ×: P=0.320 R=0.789  x: P=0.937 R=0.738  ÷: R=1.000  other14_acc=0.9821
```

P = precision, R = recall. "other14_acc" is accuracy on the 14 classes that aren't × or x.
