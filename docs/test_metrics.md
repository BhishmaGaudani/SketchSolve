# Test set results

Test images: 5064
**Test accuracy: 0.9883** (5005 / 5064 correct)

## Per-class report

```
              precision    recall  f1-score   support

           0     0.9891    0.9963    0.9927       272
           1     0.9916    0.9867    0.9891       600
           2     0.9835    0.9933    0.9884       600
           3     0.9946    0.9919    0.9932       370
           4     0.9876    0.9715    0.9795       246
           5     0.9742    1.0000    0.9869       151
           6     1.0000    1.0000    1.0000       122
           7     0.9487    0.9823    0.9652       113
           8     0.9907    0.9817    0.9862       109
           9     0.9561    0.9732    0.9646       112
           +     0.9868    0.9933    0.9900       600
           -     0.9852    0.9983    0.9917       600
           ÷     0.9200    0.9583    0.9388        24
           =     1.0000    0.9817    0.9907       545
           x     0.9983    0.9783    0.9882       600

    accuracy                         0.9883      5064
   macro avg     0.9804    0.9858    0.9830      5064
weighted avg     0.9885    0.9883    0.9884      5064

```

## Most common mistakes

- = predicted as -: 9 of 545
- x predicted as 1: 5 of 600
- 4 predicted as 9: 4 of 246
- x predicted as +: 4 of 600
- 1 predicted as 2: 4 of 600
- 7 predicted as +: 2 of 113
- 2 predicted as 7: 2 of 600
- + predicted as 7: 2 of 600
