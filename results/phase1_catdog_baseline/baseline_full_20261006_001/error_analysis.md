# Baseline validation error review

All 60 validation errors were exported to error_cases.csv and errors_1.jpg through errors_3.jpg. These are validation cases, not the unlabeled 500-image test set.

Observed examples:
- cat.10863.jpg and cat.7920.jpg contain both a cat and a dog. A single image label does not describe both objects.
- cat.10539.jpg shows only a small body detail; species cues are limited.
- cat.2939.jpg is a drawing rather than an ordinary photograph.
- dog.3994.jpg and dog.7076.jpg have substantial blank margins or composite layout. Center cropping may remove useful content; this is a hypothesis until the transformed input is inspected.
- dog.3753.jpg and dog.9109.jpg appear to depict cats in the contact sheets. They are suspected label inconsistencies, not confirmed relabeling decisions. Original labels remain unchanged.
- Other errors include clearly recognizable cats and dogs; ambiguous examples do not explain all errors.

Training evidence:
Best validation accuracy was 98.80% at epoch 2. At epoch 10 training accuracy was 98.955% and validation accuracy was 98.72%. Continued training lowered training loss without sustained validation improvement. This is consistent with a plateau and possible mild overfitting, not proof that insufficient sample count is the primary cause. Only the classification head was trained; the pretrained backbone and BatchNorm statistics were frozen.

Recommendation:
Stop extending the unchanged baseline. Inspect original-versus-preprocessed representative cases before attributing crop errors. Preserve validation labels for the official comparison; document suspicious cases separately. A limited augmentation comparison remains useful for the required data-processing discussion. Fine-tuning is conditional on time and evidence, rather than automatically necessary. No new training was started during this error review.
