# Information Box

In this folder, we provide scripts for several PEFT algorithms. We focus on their implementation, as you know, certain details are only reflected in the code and not mentioned in the papers. In each script, we primarily wrote the forward function.

LoMAP is implemented as an opt-in mode of `loralib.Linear`:

```python
layer = lora.Linear(in_features, out_features, r=8, lora_alpha=16, use_map=True)
```

The convenience alias `lora.LoMAPLinear(...)` enables the same behavior.

If you have a more complete or elegant implementation, or if you notice discrepancies in our implementation, we welcome you to submit a pull request to our repository.
