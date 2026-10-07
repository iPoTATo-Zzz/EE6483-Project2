# D1 / D2 comparison

D1 accuracy: 98.8000%

D2 accuracy: 98.8400%

Fixed errors: 16; introduced errors: 14; persistent errors: 44.

D2 training accuracy uses randomly augmented images and is not directly comparable to clean D1 training accuracy. Validation preprocessing is identical. RandomResizedCrop can use its fallback crop on extreme aspect ratios; scale bounds are sampling targets, not a guarantee for every output. This single-seed result does not establish a statistically stable improvement. D3 fine-tuning has not been executed.
