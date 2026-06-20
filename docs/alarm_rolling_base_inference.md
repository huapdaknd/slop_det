# Alarm rolling-base inference

This inference path is intended for ordered scene batches such as `data1`.

Behavior:

- Process images in scene-name order and filename order.
- After every processed image, update that scene's base to the current image.
- Keep only alarm labels in output LabelMe JSON:
  - `vegetation_loss`
  - `rock_fall`
  - `landslide`
- Clear the accepted output shapes when the accepted alarm area ratio is below `0.005`.

Example:

```bash
python tools/run_alarm_rolling_base_classifier.py \
  --current-root data1 \
  --output-root output_alarm_rolling_base_data1 \
  --expected-images 84
```

The default configs are:

- `config/model_config_rolling_base.json`
- `config/classifier_label_config_alarm_only.json`

For repeatable evaluations, restore the desired `base_data` snapshot before running,
because this path intentionally changes `base_data`.
